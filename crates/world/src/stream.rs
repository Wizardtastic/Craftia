//! Background chunk streaming: generate, mesh, load and unload chunks around a
//! moving focus point (the player). All heavy work runs off the render thread
//! via a dedicated worker thread that dispatches `rayon` parallel batches for
//! generation and meshing.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use flume::{unbounded, Receiver, Sender};
use rayon::prelude::*;

use glam::Vec3;
use voxel_core::{
    chunk_neighbours,
    math::{block_to_chunk, chunk_center, chunk_origin, world_to_block, ChunkPos},
    Frustum, CHUNK_SIZE,
};

use crate::{
    chunk::Chunk,
    far_lod::{build_far_lod_tile, FAR_TILE_CHUNKS},
    gen::TerrainGenerator,
    mesh::{ChunkMesh, ChunkMeshBundle, ChunkMesher},
    registry::BlockRegistry,
    world::World,
};

/// Radius of a chunk's bounding sphere (half its 16³ diagonal), used for the
/// conservative frustum test that orders the mesh queue.
const CHUNK_BOUND_RADIUS: f32 = CHUNK_SIZE as f32 * 0.866_025_4;

/// Tunable streaming parameters (in chunks).
#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Horizontal radius (in chunks) around the focus that must be loaded.
    pub load_radius: i32,
    /// Horizontal radius (in chunks) covered by coarse far-LOD tiles.
    /// Clamped to at least `load_radius`; 0 disables far LODs.
    pub lod_radius: i32,
    /// Chunks beyond this radius are eligible for unload.
    pub unload_radius: i32,
    /// Vertical half-band (in chunks) around the focus Y.
    pub vertical_half: i32,
    /// Max chunks to generate per tick (keeps frame hitches bounded).
    pub gen_batch: usize,
    /// Max chunks to mesh per tick.
    pub mesh_batch: usize,
    /// Max chunks to structure (place cross-chunk features) per tick.
    pub structure_batch: usize,
    /// Chunks beyond this distance (in chunks) from the focus use the GPU
    /// compute mesher instead of the CPU greedy mesher. 0 = always CPU mesh
    /// (Phase 2 disabled). Requires the renderer's `gpu_meshing` flag.
    pub gpu_mesh_distance: i32,
    /// Soft per-frame CPU budget for generation + meshing. Batch sizes adapt
    /// to the measured cost per chunk so a slow frame never queues an
    /// arbitrarily large burst.
    pub frame_budget_ms: f32,
    /// Persist terrain-only section meshes under the OS temp directory and
    /// reload them when the section content hash matches.
    pub mesh_cache: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            load_radius: 6,
            lod_radius: 32,
            unload_radius: 8,
            vertical_half: 3,
            gen_batch: 24,
            mesh_batch: 24,
            structure_batch: 8,
            gpu_mesh_distance: 0,
            frame_budget_ms: 2.5,
            mesh_cache: true,
        }
    }
}

/// Events the worker emits to the main thread. The renderer consumes
/// `MeshReady`/`Unloaded` to upload/free GPU buffers.
#[derive(Clone, Debug)]
pub enum ChunkStreamEvent {
    /// A chunk finished generating and was inserted into the world.
    Generated(ChunkPos),
    /// A chunk's mesh was (re)built and is ready to upload.
    MeshReady {
        pos: ChunkPos,
        bundle: ChunkMeshBundle,
    },
    /// Coarse (2x-downsampled) opaque LOD1 mesh for the same chunk.
    MeshLodReady {
        pos: ChunkPos,
        bundle: ChunkMeshBundle,
    },
    /// A coarse far-LOD tile. `key` is in tile coordinates; the renderer
    /// receives a synthetic chunk key derived from it. The bounds are
    /// explicit because one tile spans several normal chunks.
    FarLodReady {
        key: (i32, i32),
        bundle: ChunkMeshBundle,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
    },
    /// A far-LOD tile left the LOD radius.
    FarLodUnloaded { key: (i32, i32) },
    /// A chunk's raw voxel data is ready for GPU compute meshing (Phase 2).
    /// `voxels` is 18³ u16 BlockIds (16³ chunk + 1-voxel border from neighbours).
    /// Emitted for distant chunks when `gpu_mesh_distance` is configured.
    GpuMeshReady { pos: ChunkPos, voxels: Box<[u16]> },
    /// A chunk left the loaded set; free its GPU resources.
    Unloaded(ChunkPos),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Generating,
    Generated,
    Structuring,
    Structured,
    Meshing,
    Meshed,
}

enum Cmd {
    Focus(Vec3),
    /// Current frustum plus the camera's forward vector (mesh priority).
    View(Frustum, Vec3),
    Budget(f32),
    LoadRadius(u32),
    LodRadius(u32),
    Remesh(ChunkPos),
    Stats(std::sync::mpsc::SyncSender<StreamerStats>),
    Shutdown,
}

/// Snapshot of streamer queue depths and state counts.
#[derive(Clone, Debug, Default)]
pub struct StreamerStats {
    pub gen_queue: u32,
    pub mesh_queue: u32,
    pub pending_remesh: u32,
    pub total_tracked: u32,
    /// Wall-clock ms spent in generation batches since the last stats read.
    pub gen_ms: f32,
    /// Wall-clock ms spent in mesh batches (incl. priority remeshes) since
    /// the last stats read.
    pub mesh_ms: f32,
}

/// Owns the worker thread. Drop shuts it down.
pub struct ChunkStreamer {
    cmd_tx: Sender<Cmd>,
    pub events: Receiver<ChunkStreamEvent>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ChunkStreamer {
    /// Spawn a streamer attached to `world`. The world must outlive the streamer;
    /// the streamer holds an `Arc` clone so dropping the streamer stops touching it.
    pub fn spawn(world: Arc<World>, config: StreamConfig) -> Result<Self> {
        let (cmd_tx, cmd_rx) = unbounded();
        let (event_tx, event_rx) = unbounded();
        let handle = std::thread::Builder::new()
            .name("voxel-stream".into())
            .spawn(move || {
                run_worker(world, config, cmd_rx, event_tx);
            })
            .context("spawn stream worker thread")?;
        Ok(Self {
            cmd_tx,
            events: event_rx,
            handle: Some(handle),
        })
    }

    /// Update the streaming focus (player position in world space).
    pub fn set_focus(&self, pos: Vec3) {
        let _ = self.cmd_tx.send(Cmd::Focus(pos));
    }

    /// Update the view frustum and camera forward for mesh-building priority.
    pub fn set_frustum(&self, frustum: Frustum, forward: Vec3) {
        let _ = self.cmd_tx.send(Cmd::View(frustum, forward));
    }

    /// Update the soft CPU budget for streaming work, in milliseconds.
    pub fn set_budget(&self, budget_ms: f32) {
        let _ = self.cmd_tx.send(Cmd::Budget(budget_ms.clamp(0.25, 8.0)));
    }

    /// Change the load radius at runtime.
    pub fn set_load_radius(&self, radius: u32) {
        let _ = self.cmd_tx.send(Cmd::LoadRadius(radius));
    }

    /// Change the far-LOD radius at runtime.
    pub fn set_lod_radius(&self, radius: u32) {
        let _ = self.cmd_tx.send(Cmd::LodRadius(radius));
    }

    /// Request a remesh of a chunk (after a gameplay edit).
    pub fn request_remesh(&self, pos: ChunkPos) {
        let _ = self.cmd_tx.send(Cmd::Remesh(pos));
    }

    /// Query current streamer stats (queue depths, state counts).
    /// Returns None if the worker doesn't respond within 5ms.
    pub fn stats(&self) -> Option<StreamerStats> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let _ = self.cmd_tx.send(Cmd::Stats(tx));
        rx.recv_timeout(std::time::Duration::from_millis(5)).ok()
    }

    /// Drain any pending events without blocking.
    pub fn poll_events(&self) -> Vec<ChunkStreamEvent> {
        self.events.drain().collect()
    }
}

impl Drop for ChunkStreamer {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run_worker(
    world: Arc<World>,
    config: StreamConfig,
    cmd_rx: Receiver<Cmd>,
    event_tx: Sender<ChunkStreamEvent>,
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads())
        .build()
        .expect("rayon pool");
    let mesher = ChunkMesher;
    let section_cache: SectionCache = Arc::new(Mutex::new(HashMap::new()));
    let gen: Arc<TerrainGenerator> = world.terrain();
    let reg: Arc<BlockRegistry> = world.registry();
    let registry_fp = registry_fingerprint(&reg);
    let disk_cache = MeshDiskCache::new(config.mesh_cache, world.seed());
    let mut focus = Vec3::new(0.0, 80.0, 0.0);
    let mut desired = Vec::new();
    let mut desired_chunks = std::collections::HashSet::new();
    let mut frustum = None::<Frustum>;
    let mut view_forward = Vec3::NEG_Z;
    let mut budget_ms = config.frame_budget_ms;
    let mut gen_per_chunk_ms = 0.0f32;
    let mut mesh_per_chunk_ms = 0.0f32;
    let mut load_radius = config.load_radius;
    let mut lod_radius = config.lod_radius.max(load_radius);
    let mut unload_radius = config.unload_radius;
    let mut state: HashMap<ChunkPos, State> = HashMap::new();
    let mut far_tiles: HashMap<(i32, i32), ()> = HashMap::new();
    // Wall-clock ms spent generating / meshing since the last `Cmd::Stats`
    // read; reported in `StreamerStats` and reset there.
    let mut gen_accum_ms: f32 = 0.0;
    let mut mesh_accum_ms: f32 = 0.0;

    loop {
        // --- drain commands ---
        let mut remesh_requests: Vec<ChunkPos> = Vec::new();
        let mut shutdown = false;
        let mut focus_changed = false;
        for cmd in cmd_rx.drain() {
            match cmd {
                Cmd::Focus(p) => {
                    focus = p;
                    focus_changed = true;
                }
                Cmd::View(f, forward) => {
                    frustum = Some(f);
                    view_forward = forward;
                }
                Cmd::Budget(ms) => {
                    budget_ms = ms;
                }
                Cmd::LoadRadius(r) => {
                    load_radius = r as i32;
                    unload_radius = unload_radius.max(load_radius + 2);
                    log::info!("load radius changed to {load_radius}");
                }
                Cmd::LodRadius(r) => {
                    lod_radius = (r as i32).max(load_radius);
                    log::info!("LOD radius changed to {lod_radius}");
                }
                Cmd::Remesh(pos) => remesh_requests.push(pos),
                Cmd::Stats(tx) => {
                    let mut stats = StreamerStats::default();
                    for s in state.values() {
                        stats.total_tracked += 1;
                        match s {
                            State::Generating => stats.gen_queue += 1,
                            State::Meshing => stats.mesh_queue += 1,
                            _ => {}
                        }
                    }
                    stats.pending_remesh = remesh_requests.len() as u32;
                    stats.gen_ms = gen_accum_ms;
                    stats.mesh_ms = mesh_accum_ms;
                    gen_accum_ms = 0.0;
                    mesh_accum_ms = 0.0;
                    let _ = tx.try_send(stats);
                }
                Cmd::Shutdown => {
                    shutdown = true;
                    break;
                }
            }
        }
        if shutdown {
            return;
        }

        // Desired loaded set (cylindrical band around focus).
        let focus_chunk = block_to_chunk(world_to_block(focus));
        desired.clear();
        desired.extend(desired_set(focus_chunk, load_radius, config.vertical_half));

        // Unload chunks outside the unload radius.
        let unload_r2 = (unload_radius + 1) as i64 * (unload_radius + 1) as i64;
        let to_unload: Vec<ChunkPos> = state
            .keys()
            .filter(|p| {
                let dx = (p.x() - focus_chunk.x()) as i64;
                let dz = (p.z() - focus_chunk.z()) as i64;
                let dy = (p.y() - focus_chunk.y()) as i64;
                dx * dx + dz * dz > unload_r2 || dy.abs() > config.vertical_half as i64 + 1
            })
            .copied()
            .collect();
        for p in to_unload {
            state.remove(&p);
            section_cache.lock().remove(&p);
            world.remove_chunk(p);
            let _ = event_tx.send(ChunkStreamEvent::Unloaded(p));
        }

        // Honour direct remesh requests (e.g. from player edits).
        // Track which chunks need priority remesh (user-triggered edits).
        let mut priority_remesh_set: std::collections::HashSet<ChunkPos> =
            std::collections::HashSet::new();

        for p in remesh_requests.drain(..) {
            if let Some(s) = state.get_mut(&p) {
                if *s == State::Meshed || *s == State::Meshing {
                    *s = State::Structured; // force re-mesh without re-structuring
                    priority_remesh_set.insert(p);
                }
            }
            // Also remesh the 6 neighbours, since border faces may change.
            for n in neighbours(p) {
                if let Some(s) = state.get_mut(&n) {
                    if *s == State::Meshed {
                        *s = State::Structured;
                        priority_remesh_set.insert(n);
                    }
                }
            }
        }

        // --- PRIORITY remesh phase: immediately mesh user-triggered remeshes
        // (block break/place) before the regular distance-sorted queue. ---
        if !priority_remesh_set.is_empty() {
            let priority_remesh: Vec<ChunkPos> = priority_remesh_set.into_iter().collect();
            let world = world.clone();
            let reg = reg.clone();
            for pos in &priority_remesh {
                state.insert(*pos, State::Meshing);
            }
            let mesh_t0 = Instant::now();
            let meshes: Vec<(ChunkPos, ChunkMeshBundle, ChunkMeshBundle)> = pool.install(|| {
                priority_remesh
                    .par_iter()
                    .map(|&pos| {
                        let (bundle, lod1) = mesh_chunk_sections(
                            &world,
                            &reg,
                            &mesher,
                            &section_cache,
                            &disk_cache,
                            registry_fp,
                            pos,
                        );
                        (pos, bundle, lod1)
                    })
                    .collect()
            });
            let remesh_elapsed = mesh_t0.elapsed().as_secs_f32() * 1000.0;
            mesh_accum_ms += remesh_elapsed;
            if !meshes.is_empty() {
                mesh_per_chunk_ms = ema(
                    mesh_per_chunk_ms,
                    remesh_elapsed / meshes.len() as f32,
                );
            }
            for (pos, bundle, lod1) in meshes {
                world.insert_mesh(pos);
                state.insert(pos, State::Meshed);
                let _ = event_tx.send(ChunkStreamEvent::MeshReady { pos, bundle });
                if !lod1.is_empty() {
                    let _ = event_tx.send(ChunkStreamEvent::MeshLodReady { pos, bundle: lod1 });
                }
            }
        }

        // --- generation phase ---
        let gen_todo: Vec<ChunkPos> = desired
            .iter()
            .filter(|p| !state.contains_key(p))
            .copied()
            .collect();
        let gen_todo = sort_by_distance(gen_todo, focus_chunk);
        let gen_cap = adaptive_cap(config.gen_batch, gen_per_chunk_ms, budget_ms);
        let gen_batch: Vec<ChunkPos> = gen_todo
            .into_iter()
            .take(gen_cap)
            .inspect(|p| {
                state.insert(*p, State::Generating);
            })
            .collect();

        if !gen_batch.is_empty() {
            let gen = gen.clone();
            let reg = reg.clone();
            let world_for_light = world.clone();
            let gen_t0 = Instant::now();
            let generated: Vec<(ChunkPos, Chunk)> = pool.install(|| {
                gen_batch
                    .par_iter()
                    .filter_map(|&pos| {
                        let mut chunk = Chunk::new(pos);
                        let columns = gen.column_table(chunk_origin(pos));
                        if gen.generate(&mut chunk, &reg, &columns).is_err() {
                            log::error!("chunk generation failed for {pos:?}");
                            return None;
                        }
                        gen.decorate(
                            &mut chunk,
                            &reg,
                            &columns,
                            |wx, wy, wz| world_for_light.get_block(wx, wy, wz),
                            |wx, wy, wz, id| world_for_light.set_block_no_light(wx, wy, wz, id),
                        );
                        Some((pos, chunk))
                    })
                    .collect()
            });
            let gen_elapsed = gen_t0.elapsed().as_secs_f32() * 1000.0;
            gen_accum_ms += gen_elapsed;
            if !generated.is_empty() {
                gen_per_chunk_ms = ema(
                    gen_per_chunk_ms,
                    gen_elapsed / generated.len() as f32,
                );
            }
            for (pos, chunk) in generated {
                world.insert_chunk(pos, chunk);
                state.insert(pos, State::Generated);
                let _ = event_tx.send(ChunkStreamEvent::Generated(pos));
                // Remesh loaded neighbours so their border faces update now
                // that this chunk exists. They don't need to re-run structures.
                for n in neighbours(pos) {
                    if let Some(s) = state.get_mut(&n) {
                        if *s == State::Meshed {
                            *s = State::Structured;
                        }
                    }
                }
            }
        }

        // --- structure phase: place cross-chunk features (dungeons, towers,
        // wells) on chunks that have finished generating. A chunk can only be
        // structured when its 8 horizontal neighbours are also ready
        // (Generated / Structuring / Structured / Meshed), or are outside the
        // desired loaded set (edge of load radius). This guarantees the
        // `sample` closure reads real neighbour data when verifying structure
        // placement conditions (e.g. "is the surface block grass?"). ---
        desired_chunks.clear();
        desired_chunks.extend(desired.iter().copied());
        let struct_todo: Vec<ChunkPos> = desired
            .iter()
            .filter(|p| matches!(state.get(p), Some(State::Generated)))
            .filter(|p| {
                horizontal_neighbours(**p)
                    .iter()
                    .all(|n| match state.get(n) {
                        Some(State::Generated)
                        | Some(State::Structuring)
                        | Some(State::Structured)
                        | Some(State::Meshed) => true,
                        _ => !desired_chunks.contains(n),
                    })
            })
            .copied()
            .collect();
        let struct_todo = sort_by_distance(struct_todo, focus_chunk);
        let struct_batch: Vec<ChunkPos> = struct_todo
            .into_iter()
            .take(config.structure_batch)
            .inspect(|p| {
                state.insert(*p, State::Structuring);
            })
            .collect();

        if !struct_batch.is_empty() {
            let gen = gen.clone();
            let reg = reg.clone();
            let world_for_sample = world.clone();
            let structured: Vec<(ChunkPos, Chunk)> = pool.install(|| {
                struct_batch
                    .par_iter()
                    .map(|&pos| {
                        // Clone the chunk out from under the lock so the
                        // worker can mutate it while the `sample` closure
                        // re-acquires a read lock for cross-chunk queries.
                        let chunk_clone = {
                            let chunks = world_for_sample.chunks.read_shard(pos);
                            chunks.get(&pos).cloned()
                        };
                        let mut chunk = chunk_clone.unwrap_or_else(|| Chunk::new(pos));
                        gen.place_structures(&mut chunk, &reg, &|x, y, z| {
                            world_for_sample.get_block(x, y, z)
                        });
                        chunk.dirty = true;
                        (pos, chunk)
                    })
                    .collect()
            });
            for (pos, chunk) in structured {
                world.insert_chunk(pos, chunk);
                state.insert(pos, State::Structured);
                let _ = event_tx.send(ChunkStreamEvent::Generated(pos));
                // Demote Meshed neighbours so their border faces get rebuilt
                // (a structure may have spilled across the boundary).
                for n in neighbours(pos) {
                    if let Some(s) = state.get_mut(&n) {
                        if *s == State::Meshed {
                            *s = State::Structured;
                        }
                    }
                }
            }
        }

        // --- meshing phase: chunks that are structured and have structured
        // neighbours (so border face culling is correct enough).
        // Every structured chunk is eventually meshed; the frustum only
        // orders the queue so the currently visible near field fills first.
        // (Skipping off-screen chunks left holes that popped in when the
        // player turned, which is why this is priority-only.)
        let mut mesh_todo: Vec<ChunkPos> = desired
            .iter()
            .filter(|p| matches!(state.get(p), Some(State::Structured)))
            .copied()
            .collect();
        mesh_todo.sort_by_key(|p| {
            // Conservative bounding-sphere test: much cheaper than building
            // an AABB per chunk and good enough for queue ordering.
            let center = chunk_center(*p);
            let visible = frustum.as_ref().map_or(true, |f| {
                f.intersects_sphere(center, CHUNK_BOUND_RADIUS)
            });
            // Screen-space area ~ radius² / distance², weighted by how close
            // the chunk is to the view axis. Bucket the facing term so tiny
            // orientation changes don't reshuffle the queue every frame.
            let facing = (center - focus).normalize_or_zero().dot(view_forward);
            let angle_bucket = if facing > 0.75 {
                0u8
            } else if facing > 0.25 {
                1
            } else {
                2
            };
            let dx = p.x() - focus_chunk.x();
            let dy = p.y() - focus_chunk.y();
            let dz = p.z() - focus_chunk.z();
            (u8::from(!visible), angle_bucket, dx * dx + dy * dy + dz * dz)
        });

        // Empty chunks (air above the terrain, mostly) have no faces to build.
        // Mark them meshed and move on: an empty bundle is a no-op upload, so
        // skipping the build costs nothing but saves a full registry scan.
        let (empty, mesh_todo): (Vec<ChunkPos>, Vec<ChunkPos>) = mesh_todo
            .into_iter()
            .partition(|p| world.chunk_non_air(*p) == 0);
        // A fully sealed underground chunk (solid itself and opaque on all
        // six neighbouring border layers) cannot contribute a single face.
        // Checking it costs ~5.5k block reads versus a full greedy mesh, and
        // the common deep-terrain case skips thousands of these per session.
        let (enclosed, mesh_todo): (Vec<ChunkPos>, Vec<ChunkPos>) = mesh_todo
            .into_iter()
            .partition(|p| world.chunk_is_enclosed(*p));
        for pos in empty.into_iter().chain(enclosed) {
            world.insert_mesh(pos);
            state.insert(pos, State::Meshed);
            let _ = event_tx.send(ChunkStreamEvent::MeshReady {
                pos,
                bundle: ChunkMeshBundle::default(),
            });
        }

        let mesh_cap = adaptive_cap(config.mesh_batch, mesh_per_chunk_ms, budget_ms);
        let mesh_batch: Vec<ChunkPos> = mesh_todo
            .into_iter()
            .take(mesh_cap)
            .inspect(|p| {
                state.insert(*p, State::Meshing);
            })
            .collect();

        if !mesh_batch.is_empty() {
            // Split the batch into near (CPU greedy mesher) and far (GPU
            // compute mesher, Phase 2). With `gpu_mesh_distance <= 0` every
            // chunk is "near" and the GPU path never engages.
            let gpu_dist = config.gpu_mesh_distance.max(0) as i64;
            let (near, far): (Vec<ChunkPos>, Vec<ChunkPos>) =
                mesh_batch.iter().copied().partition(|p| {
                    // The GPU compute mesher emits static geometry and cannot
                    // evaluate the wave field, so any chunk holding water must
                    // stay on the CPU mesher.
                    if world.chunk_needs_cpu_mesher(*p) {
                        return true;
                    }
                    if config.gpu_mesh_distance <= 0 {
                        return true;
                    }
                    let dx = (p.x() - focus_chunk.x()) as i64;
                    let dy = (p.y() - focus_chunk.y()) as i64;
                    let dz = (p.z() - focus_chunk.z()) as i64;
                    dx * dx + dy * dy + dz * dz <= gpu_dist * gpu_dist
                });

            // --- GPU meshing for distant chunks (Phase 2) ---
            if !far.is_empty() {
                for pos in &far {
                    let voxels = world.gpu_mesh_voxels(*pos);
                    world.insert_mesh(*pos);
                    state.insert(*pos, State::Meshed);
                    let _ = event_tx.send(ChunkStreamEvent::GpuMeshReady { pos: *pos, voxels });
                }
            }

            // --- CPU meshing for near chunks ---
            if !near.is_empty() {
                let world = world.clone();
                let reg = reg.clone();
                let mesher = &mesher;
                    let mesh_t0 = Instant::now();
                let meshes: Vec<(ChunkPos, ChunkMeshBundle, ChunkMeshBundle)> = pool.install(|| {
                    near.par_iter()
                        .map(|&pos| {
                            let (bundle, lod1) = mesh_chunk_sections(
                                &world,
                                &reg,
                                mesher,
                                &section_cache,
                                &disk_cache,
                                registry_fp,
                                pos,
                            );
                            (pos, bundle, lod1)
                        })
                        .collect()
                });
                mesh_accum_ms += mesh_t0.elapsed().as_secs_f32() * 1000.0;
                for (pos, bundle, lod1) in meshes {
                    world.insert_mesh(pos);
                    state.insert(pos, State::Meshed);
                    let _ = event_tx.send(ChunkStreamEvent::MeshReady { pos, bundle });
                    if !lod1.is_empty() {
                        let _ = event_tx.send(ChunkStreamEvent::MeshLodReady { pos, bundle: lod1 });
                    }
                }
            }
        }

        // --- Far LOD ring ------------------------------------------------
        // Tiles are independent of the detailed load radius once generated:
        // a tile that is partly under full chunks simply gets depth-tested
        // behind them. That lets the standard radius slider move without
        // rebuilding the far world.
        let tile_chunks = FAR_TILE_CHUNKS;
        let tile_blocks = (tile_chunks * CHUNK_SIZE) as f32;
        let base_blocks = load_radius as f32 * CHUNK_SIZE as f32;
        let lod_blocks = lod_radius as f32 * CHUNK_SIZE as f32;
        let tile_r = (lod_blocks / tile_blocks).ceil() as i32 + 1;
        let center_tx = focus_chunk.x().div_euclid(tile_chunks);
        let center_tz = focus_chunk.z().div_euclid(tile_chunks);
        let mut desired_far: std::collections::HashSet<(i32, i32)> =
            std::collections::HashSet::new();
        if lod_radius > load_radius {
            for tz in (center_tz - tile_r)..=(center_tz + tile_r) {
                for tx in (center_tx - tile_r)..=(center_tx + tile_r) {
                    let ox = (tx * tile_chunks * CHUNK_SIZE) as f32;
                    let oz = (tz * tile_chunks * CHUNK_SIZE) as f32;
                    let cx = ox + tile_blocks * 0.5;
                    let cz = oz + tile_blocks * 0.5;
                    let dx = cx - focus.x;
                    let dz = cz - focus.z;
                    let half_diag = tile_blocks * 0.7071;
                    if (dx * dx + dz * dz).sqrt() > lod_blocks + half_diag + 16.0 {
                        continue;
                    }
                    // Fully covered by full-detail chunks: no need to draw a
                    // duplicate coarse surface underneath them.
                    let mut far_corner = 0.0f32;
                    for (px, pz) in [
                        (ox, oz),
                        (ox + tile_blocks, oz),
                        (ox, oz + tile_blocks),
                        (ox + tile_blocks, oz + tile_blocks),
                    ] {
                        far_corner = far_corner.max(
                            ((px - focus.x).powi(2) + (pz - focus.z).powi(2)).sqrt(),
                        );
                    }
                    if far_corner < (base_blocks - 8.0).max(0.0) {
                        continue;
                    }
                    desired_far.insert((tx, tz));
                }
            }
        }
        let stale: Vec<(i32, i32)> = far_tiles
            .keys()
            .filter(|k| !desired_far.contains(k))
            .copied()
            .collect();
        for key in stale {
            far_tiles.remove(&key);
            let _ = event_tx.send(ChunkStreamEvent::FarLodUnloaded { key });
        }
        let mut missing: Vec<(i32, i32)> = desired_far
            .into_iter()
            .filter(|k| !far_tiles.contains_key(k))
            .collect();
        if !missing.is_empty() {
            missing.sort_by_key(|(tx, tz)| {
                let cx = (*tx * tile_chunks * CHUNK_SIZE) as f32 + tile_blocks * 0.5;
                let cz = (*tz * tile_chunks * CHUNK_SIZE) as f32 + tile_blocks * 0.5;
                ((cx - focus.x).powi(2) + (cz - focus.z).powi(2)) as i64
            });
            missing.truncate(8);
            let gen = gen.clone();
            let reg = reg.clone();
            let built: Vec<((i32, i32), crate::far_lod::FarLodTile)> = pool.install(|| {
                missing
                    .par_iter()
                    .map(|&(tx, tz)| ((tx, tz), build_far_lod_tile(&gen, &reg, tx, tz)))
                    .collect()
            });
            for (key, tile) in built {
                far_tiles.insert(key, ());
                let _ = event_tx.send(ChunkStreamEvent::FarLodReady {
                    key,
                    bundle: ChunkMeshBundle {
                        opaque: tile.opaque,
                        transparent: tile.transparent,
                    },
                    bounds_min: tile.bounds_min,
                    bounds_max: tile.bounds_max,
                });
            }
        }

        // If nothing happened this iteration and the focus didn't just move,
        // sleep briefly to avoid a busy polling loop. Commands stay queued in
        // the channel and are handled by the next iteration's drain — there
        // is exactly one command-processing path.
        if gen_batch.is_empty()
            && struct_batch.is_empty()
            && mesh_batch.is_empty()
            && !focus_changed
            && cmd_rx.is_empty()
        {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

const MESH_CACHE_VERSION: u32 = 2;
const MESH_CACHE_MAGIC: &[u8; 4] = b"VXMS";

/// A small versioned binary cache for one section mesh. Files are keyed by
/// chunk + section; the header carries a content hash so stale files are
/// rejected rather than trusted.
struct MeshDiskCache {
    root: Option<std::path::PathBuf>,
}

impl MeshDiskCache {
    fn new(enabled: bool, seed: i32) -> Self {
        let root = enabled.then(|| {
            std::env::temp_dir()
                .join("voxel-engine-mesh-cache")
                .join(format!("{seed}"))
        });
        Self { root }
    }

    fn enabled(&self) -> bool {
        self.root.is_some()
    }

    fn path(&self, pos: ChunkPos, section: u8) -> Option<std::path::PathBuf> {
        Some(self.root.as_ref()?.join(format!(
            "{}_{}_{}_{}.mesh",
            pos.x(),
            pos.y(),
            pos.z(),
            section
        )))
    }

    fn read(&self, pos: ChunkPos, section: u8, hash: u64) -> Option<ChunkMeshBundle> {
        let path = self.path(pos, section)?;
        let data = std::fs::read(path).ok()?;
        if data.len() < 32 || &data[0..4] != MESH_CACHE_MAGIC {
            return None;
        }
        let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
        let stored = u64::from_le_bytes(data[8..16].try_into().ok()?);
        if version != MESH_CACHE_VERSION || stored != hash {
            return None;
        }
        let ov = u32::from_le_bytes(data[16..20].try_into().ok()?) as usize;
        let oi = u32::from_le_bytes(data[20..24].try_into().ok()?) as usize;
        let tv = u32::from_le_bytes(data[24..28].try_into().ok()?) as usize;
        let ti = u32::from_le_bytes(data[28..32].try_into().ok()?) as usize;
        const V: usize = std::mem::size_of::<crate::mesh::ChunkVertex>();
        const I: usize = std::mem::size_of::<u32>();
        let need = 32 + ov * V + oi * I + tv * V + ti * I;
        if data.len() != need || ov + tv > 4_000_000 || oi + ti > 8_000_000 {
            return None;
        }
        let mut off = 32;
        let opaque_v: Vec<crate::mesh::ChunkVertex> = data[off..off + ov * V]
            .chunks_exact(V)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        off += ov * V;
        let opaque_i: Vec<u32> = data[off..off + oi * I]
            .chunks_exact(I)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        off += oi * I;
        let trans_v: Vec<crate::mesh::ChunkVertex> = data[off..off + tv * V]
            .chunks_exact(V)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        off += tv * V;
        let trans_i: Vec<u32> = data[off..off + ti * I]
            .chunks_exact(I)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Some(ChunkMeshBundle {
            opaque: ChunkMesh {
                vertices: opaque_v,
                indices: opaque_i,
            },
            transparent: ChunkMesh {
                vertices: trans_v,
                indices: trans_i,
            },
        })
    }

    fn write(&self, pos: ChunkPos, section: u8, hash: u64, bundle: &ChunkMeshBundle) {
        // Empty sections are cheaper to rebuild than to hit the filesystem for.
        if bundle.is_empty() {
            return;
        }
        let Some(path) = self.path(pos, section) else {
            return;
        };
        let Some(dir) = path.parent() else {
            return;
        };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        let ov = bundle.opaque.vertices.len() as u32;
        let oi = bundle.opaque.indices.len() as u32;
        let tv = bundle.transparent.vertices.len() as u32;
        let ti = bundle.transparent.indices.len() as u32;
        let mut data = Vec::with_capacity(
            32 + (ov as usize + tv as usize) * std::mem::size_of::<crate::mesh::ChunkVertex>()
                + (oi as usize + ti as usize) * 4,
        );
        data.extend_from_slice(MESH_CACHE_MAGIC);
        data.extend_from_slice(&MESH_CACHE_VERSION.to_le_bytes());
        data.extend_from_slice(&hash.to_le_bytes());
        data.extend_from_slice(&ov.to_le_bytes());
        data.extend_from_slice(&oi.to_le_bytes());
        data.extend_from_slice(&tv.to_le_bytes());
        data.extend_from_slice(&ti.to_le_bytes());
        data.extend_from_slice(bytemuck::cast_slice(&bundle.opaque.vertices));
        data.extend_from_slice(bytemuck::cast_slice(&bundle.opaque.indices));
        data.extend_from_slice(bytemuck::cast_slice(&bundle.transparent.vertices));
        data.extend_from_slice(bytemuck::cast_slice(&bundle.transparent.indices));
        static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = path.with_extension(format!("tmp{}_{}", std::process::id(), seq));
        if std::fs::write(&tmp, &data).is_err() {
            return;
        }
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::rename(&tmp, &path);
    }
}

fn registry_fingerprint(reg: &BlockRegistry) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    reg.len().hash(&mut h);
    for i in 0..reg.len() {
        let def = reg.get(voxel_core::BlockId(i as u16));
        def.opaque.hash(&mut h);
        (def.kind as u8).hash(&mut h);
        def.textures.tiles.hash(&mut h);
        def.material.flags.hash(&mut h);
        def.is_rendered().hash(&mut h);
    }
    h.finish()
}

#[derive(Default)]
struct SectionCacheEntry {
    sections: [Option<ChunkMeshBundle>; 8],
    lod1: Option<ChunkMeshBundle>,
}

type SectionEntry = Arc<Mutex<SectionCacheEntry>>;
type SectionCache = Arc<Mutex<HashMap<ChunkPos, SectionEntry>>>;

fn concat_sections(entry: &SectionCacheEntry) -> ChunkMeshBundle {
    let mut out = ChunkMeshBundle::default();
    for section in entry.sections.iter().flatten() {
        out.opaque.append(&section.opaque);
        out.transparent.append(&section.transparent);
    }
    out
}

/// Mesh one chunk from its 8 cached sections. Only sections listed in the
/// world's dirty mask are rebuilt; the rest are reused verbatim. A missing
/// cache entry forces all 8 sections on the first mesh.
fn mesh_chunk_sections(
    world: &World,
    reg: &BlockRegistry,
    mesher: &ChunkMesher,
    cache: &SectionCache,
    disk: &MeshDiskCache,
    registry_fp: u64,
    pos: ChunkPos,
) -> (ChunkMeshBundle, ChunkMeshBundle) {
    let mut dirty = world.take_dirty_sections(pos).unwrap_or(0xFF);
    // Static terrain, foliage and cacti are cacheable; only liquid surfaces
    // depend on dynamic water-column metadata that the block hash misses.
    let cacheable = disk.enabled() && !world.chunk_has_water(pos);
    // Take only the per-chunk entry lock; the map lock is released first so
    // unrelated chunks can mesh in parallel.
    let entry: SectionEntry = {
        let mut map = cache.lock();
        map.entry(pos).or_default().clone()
    };
    let mut entry = entry.lock();
    let initial = entry.sections.iter().all(|s| s.is_none());
    if initial {
        dirty = 0xFF;
    }
    for section in 0..8u8 {
        if dirty & (1 << section) == 0 {
            continue;
        }
        let hash = if cacheable {
            Some(world.section_cache_hash(pos, section, registry_fp))
        } else {
            None
        };
        if initial {
            if let Some(hash) = hash {
                if let Some(bundle) = disk.read(pos, section, hash) {
                    entry.sections[section as usize] = Some(bundle);
                    continue;
                }
            }
        }
        let bundle = world.with_chunk_for_mesh(pos, |chunk, sample, sample_water, sample_loaded| {
            mesher.build_section(
                section,
                chunk,
                reg,
                &sample,
                &sample_water,
                &sample_loaded,
            )
        });
        if let Some(hash) = hash {
            disk.write(pos, section, hash, &bundle);
        }
        entry.sections[section as usize] = Some(bundle);
    }
    let bundle = concat_sections(&entry);
    // LOD1 covers the whole chunk, so it is refreshed whenever any section
    // changed. It is cheap (8³ cells) compared with the section meshes.
    if dirty != 0 || entry.lod1.is_none() {
        entry.lod1 = Some(world.with_chunk_for_mesh(
            pos,
            |chunk, sample, sample_water, sample_loaded| {
                mesher.build_lod1(chunk, reg, &sample, &sample_water, &sample_loaded)
            },
        ));
    }
    let lod1 = entry.lod1.clone().unwrap_or_default();
    (bundle, lod1)
}

/// Leave one core for the render/main thread; rayon batches otherwise
/// compete with the frame that is waiting on them.
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| if n.get() > 2 { n.get() - 1 } else { n.get() })
        .unwrap_or(4)
        .max(1)
}

/// Batch size that fits `budget_ms` at the measured per-chunk cost, capped by
/// the configured maximum. Before the first measurement, use the full cap.
fn adaptive_cap(max: usize, ema_ms: f32, budget_ms: f32) -> usize {
    if max == 0 || ema_ms <= 0.01 {
        return max;
    }
    ((budget_ms / ema_ms).ceil() as usize).clamp(1, max)
}

fn ema(prev: f32, sample: f32) -> f32 {
    if prev <= 0.0 {
        sample
    } else {
        prev * 0.8 + sample * 0.2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ChunkVertex;

    #[test]
    fn disk_cache_roundtrips_and_rejects_stale_hash() {
        let dir = std::env::temp_dir().join(format!(
            "vxmc-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let disk = MeshDiskCache {
            root: Some(dir.clone()),
        };
        let pos = ChunkPos::new(3, 4, 5);
        let mut bundle = ChunkMeshBundle::default();
        bundle
            .opaque
            .vertices
            .push(ChunkVertex::new([1.0, 2.0, 3.0], [0.0, 1.0], 7, 0));
        bundle.opaque.indices.extend_from_slice(&[0, 0, 0]);
        disk.write(pos, 2, 0x1234, &bundle);
        let loaded = disk.read(pos, 2, 0x1234).expect("cache hit");
        assert_eq!(loaded.opaque.vertices.len(), bundle.opaque.vertices.len());
        assert_eq!(
            bytemuck::bytes_of(&loaded.opaque.vertices[0]),
            bytemuck::bytes_of(&bundle.opaque.vertices[0])
        );
        assert_eq!(loaded.opaque.indices, bundle.opaque.indices);
        assert!(disk.read(pos, 2, 0x9999).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn section_cache_rebuilds_after_edit() {
        let world = World::new(7);
        let pos = ChunkPos::new(0, 0, 0);
        let stone = world.registry().id_of("stone").unwrap();
        let mut chunk = Chunk::new(pos);
        for z in 0..16 {
            for x in 0..16 {
                chunk.set(x, 0, z, stone);
            }
        }
        world.insert_chunk(pos, chunk);
        let cache: SectionCache = Arc::new(Mutex::new(HashMap::new()));
        let reg = world.registry();
        let disk = MeshDiskCache { root: None };
        let fp = registry_fingerprint(&reg);
        let (before, _) = mesh_chunk_sections(
            &world, &reg, &ChunkMesher, &cache, &disk, fp, pos,
        );
        assert!(!before.opaque.is_empty());
        world.set_block(8, 0, 8, voxel_core::BlockId::AIR);
        let (after, _) = mesh_chunk_sections(
            &world, &reg, &ChunkMesher, &cache, &disk, fp, pos,
        );
        assert!(!after.opaque.is_empty());
        assert_ne!(
            before.opaque.vertices.len(),
            after.opaque.vertices.len(),
            "the edited section must be rebuilt"
        );
    }
}

/// The cylindrical band of chunks around `focus` that should be loaded.
fn desired_set(focus: ChunkPos, load_radius: i32, vertical_half: i32) -> Vec<ChunkPos> {
    let mut out = Vec::new();
    let r = load_radius;
    for dx in -r..=r {
        for dz in -r..=r {
            if dx * dx + dz * dz > r * r {
                continue;
            }
            for dy in -vertical_half..=vertical_half {
                let p = ChunkPos::new(focus.x() + dx, focus.y() + dy, focus.z() + dz);
                if p.y() < 0 || p.y() >= voxel_core::MAX_CHUNK_Y {
                    continue;
                }
                out.push(p);
            }
        }
    }
    out
}

/// The 8 XZ-plane neighbours of a chunk (used for the structure phase's
/// readiness gate — vertical neighbours are not required).
fn horizontal_neighbours(p: ChunkPos) -> [ChunkPos; 8] {
    [
        ChunkPos::new(p.x() - 1, p.y(), p.z() - 1),
        ChunkPos::new(p.x(), p.y(), p.z() - 1),
        ChunkPos::new(p.x() + 1, p.y(), p.z() - 1),
        ChunkPos::new(p.x() - 1, p.y(), p.z()),
        ChunkPos::new(p.x() + 1, p.y(), p.z()),
        ChunkPos::new(p.x() - 1, p.y(), p.z() + 1),
        ChunkPos::new(p.x(), p.y(), p.z() + 1),
        ChunkPos::new(p.x() + 1, p.y(), p.z() + 1),
    ]
}

fn neighbours(p: ChunkPos) -> [ChunkPos; 6] {
    chunk_neighbours(p)
}

fn sort_by_distance(mut v: Vec<ChunkPos>, focus: ChunkPos) -> Vec<ChunkPos> {
    v.sort_by_key(|p| {
        let dx = p.x() - focus.x();
        let dy = p.y() - focus.y();
        let dz = p.z() - focus.z();
        dx * dx + dy * dy + dz * dz
    });
    v
}
