//! Named, persistent block snapshots (**schematics**) and their
//! rotation/mirror transforms.
//!
//! Feature 3 of the AI-authoring surface (see `plans/voxel-engine-refactor.md`).
//! A [`Schematic`] is a bounded snapshot of a `World` region, keyed by a
//! [`SchematicId`] (string). Voxels store a *palette index* (u16) into a
//! text-name palette so the same `.schem` file can be loaded against any
//! [`BlockRegistry`] — only the names need to resolve.
//!
//! On-disk format (matches the conventions of `crates/world/src/save.rs`):
//! 1. **Magic**: `b"SCHEMATIC\0"` (9 bytes).
//! 2. **Version**: `u32` little-endian (current = 1).
//! 3. **Origin**: 3 × `i32` LE (minimum corner of the captured region).
//! 4. **Size**:   3 × `u32` LE (X, Y, Z in blocks).
//! 5. Inflate the rest (deflate-compressed):
//!    * **Palette length**: `u16` LE.
//!    * **Palette entries**: for each, `u16` LE name-length + name bytes.
//!    * **Voxel count**:    `u32` LE.
//!    * **Voxel entries**:  for each, `u16` LE rel_x + `u16` LE rel_y +
//!      `u16` LE rel_z + `u16` LE palette_idx; entries are sorted by
//!      `(rel_y, rel_z, rel_x)` for determinism.
//!
//! Air blocks are not stored — the palette is allowed to contain `"air"`
//! but the voxel list contains only non-air writes.
//!
//! Rotation is around the Y axis only (`Rotation90::Deg0 | Deg90 | Deg180 |
//! Deg270`). Mirror is a bitmask of X/Y/Z (`MirrorAxes(0..7)`). Both are
//! applied in this order: rotate first, then mirror.
//!
//! Internally the voxel map is `BTreeMap<(i32, i32, i32), u16>` (a tuple of
//! `i32`s, NOT `glam::IVec3`) because the public-facing key needs `Ord`
//! for the sorted save format and `glam::IVec3` does not implement `Ord`
//! in the version pinned here.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use glam::IVec3;
use voxel_core::math::Vec3f;
use voxel_core::{Aabb, BlockId};

use crate::volume::BlockChange;
use crate::{BlockRegistry, World};

/// Stable on-disk format version. Bump when the wire format in this file
/// changes (so future readers can refuse old schematics).
pub const SCHEMATIC_FORMAT_VERSION: u32 = 1;

/// File-header magic written by [`Schematic::save`] and verified by
/// [`Schematic::load`]. Kept as a single source of truth (instead of two
/// inline `b"SCHEMATIC"` literals) so a future edit cannot drift the
/// write-side and read-side back into the original 9-vs-10-byte mismatch
/// that landed once during early Feature 3 development.
pub const SCHEMATIC_MAGIC: &[u8] = b"SCHEMATIC";

/// String identifier for a named schematic. Comparable by value; cheap to
/// clone (one allocation) and `Display`-able for log/UI lines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SchematicId(pub String);

impl SchematicId {
    pub fn new(name: impl Into<String>) -> Self {
        SchematicId(name.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SchematicId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SchematicId {
    fn from(s: &str) -> Self {
        SchematicId(s.to_string())
    }
}
impl From<String> for SchematicId {
    fn from(s: String) -> Self {
        SchematicId(s)
    }
}

/// Y-axis rotation in 90° increments. v1 only — other axes are not yet
/// implemented (callers can rotate by re-positioning if needed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation90 {
    Deg0,
    Deg90,
    Deg180,
    Deg270,
}

impl Rotation90 {
    /// All four rotations in canonical order. Useful for enumerating
    /// previews or building a rotating animation tick list.
    pub const ALL: [Rotation90; 4] = [
        Rotation90::Deg0,
        Rotation90::Deg90,
        Rotation90::Deg180,
        Rotation90::Deg270,
    ];

    /// Apply this rotation to a relative voxel position. Y is unchanged.
    pub fn apply(self, p: IVec3) -> IVec3 {
        match self {
            Rotation90::Deg0 => p,
            Rotation90::Deg90 => IVec3::new(-p.z, p.y, p.x),
            Rotation90::Deg180 => IVec3::new(-p.x, p.y, -p.z),
            Rotation90::Deg270 => IVec3::new(p.z, p.y, -p.x),
        }
    }
}

/// Bitmask of axes to mirror. Bits: 0 = X, 1 = Y, 2 = Z.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MirrorAxes(pub u8);

impl MirrorAxes {
    pub const NONE: MirrorAxes = MirrorAxes(0);
    pub const X: MirrorAxes = MirrorAxes(1);
    pub const Y: MirrorAxes = MirrorAxes(2);
    pub const Z: MirrorAxes = MirrorAxes(4);
    pub const XY: MirrorAxes = MirrorAxes(3);
    pub const XZ: MirrorAxes = MirrorAxes(5);
    pub const YZ: MirrorAxes = MirrorAxes(6);
    pub const ALL: MirrorAxes = MirrorAxes(7);

    pub fn contains(self, other: MirrorAxes) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Apply this mirror to a relative voxel position. Each set axis
    /// negates that component. The "edge" of the mirror (size - 1 in that
    /// axis) is supplied by the caller via `size_minus_one` so the mirror
    /// is centered on the bounding box (POS X ↔ MAX X), not on the origin.
    pub fn apply(self, p: IVec3, size_minus_one: IVec3) -> IVec3 {
        let mx = if self.contains(MirrorAxes::X) {
            size_minus_one.x - p.x
        } else {
            p.x
        };
        let my = if self.contains(MirrorAxes::Y) {
            size_minus_one.y - p.y
        } else {
            p.y
        };
        let mz = if self.contains(MirrorAxes::Z) {
            size_minus_one.z - p.z
        } else {
            p.z
        };
        IVec3::new(mx, my, mz)
    }
}

/// Convert a `(i32, i32, i32)` tuple into an `IVec3` (used between
/// BTreeMap key reads and the rotation/mirror pathway).
///
/// `IVec3 ↔ (i32, i32, i32)` cannot have local `From` impls (orphan
/// rule — both types are foreign), so we use a free fn. Reverse
/// direction is destructured inline at usage sites.
#[inline]
pub fn pos_to_iv3(t: (i32, i32, i32)) -> IVec3 {
    IVec3::new(t.0, t.1, t.2)
}

/// One named, snapshotted region of blocks.
#[derive(Clone, Debug)]
pub struct Schematic {
    /// Identifier human callers refer to. Not persisted as part of the
    /// binary (the on-disk name is up to the file naming convention).
    pub id: SchematicId,
    /// Bounding-box of the captured region in world coordinates
    /// (post-rotation/mirror pass *not* applied — always the *original*
    /// capture box).
    pub bounds: Aabb,
    /// Block names indexed by internal `u16` palette id (0 always means
    /// AIR placeholder even if no voxel references it).
    pub palette: Vec<String>,
    /// Sparse voxel map: local position (relative to `bounds.min`,
    /// stored as `(i32, i32, i32)` for `Ord`) → palette index. Air
    /// entries are never inserted.
    voxels: BTreeMap<(i32, i32, i32), u16>,
}

impl Schematic {
    /// Capture all non-air blocks in `bounds` from `world`, using
    /// `registry` to translate numeric ids to stable text names.
    pub fn capture(id: SchematicId, bounds: Aabb, world: &World, registry: &BlockRegistry) -> Self {
        let mut palette: Vec<String> = vec!["air".to_string()];
        let mut idx: HashMap<String, u16> = HashMap::new();
        idx.insert("air".to_string(), 0);
        let mut voxels = BTreeMap::new();
        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_range(bounds);
        for y in min_y..max_y {
            for z in min_z..max_z {
                for x in min_x..max_x {
                    let block_id = world.get_block(x, y, z);
                    if block_id.is_air() {
                        continue;
                    }
                    let name = registry.get(block_id).name.to_string();
                    let pal_idx = *idx.entry(name.clone()).or_insert_with(|| {
                        let pi = palette.len() as u16;
                        palette.push(name);
                        pi
                    });
                    voxels.insert((x - min_x, y - min_y, z - min_z), pal_idx);
                }
            }
        }
        Schematic {
            id,
            bounds,
            palette,
            voxels,
        }
    }

    /// How many voxels are stored (non-air).
    pub fn voxel_count(&self) -> usize {
        self.voxels.len()
    }

    /// Iterate `(local_pos, palette_idx)` in lexicographic order
    /// equivalent to `(x, y, z)` priority. BTreeMap already provides this.
    pub fn voxels(&self) -> impl Iterator<Item = (IVec3, u16)> + '_ {
        self.voxels
            .iter()
            .map(|((x, y, z), idx)| (IVec3::new(*x, *y, *z), *idx))
    }

    /// Total block volume of the original capture region (including air).
    /// Note: bounded to a `u32` so very large captures (>= 2²¹ blocks on
    /// one axis, ~2 GiB) would overflow — v1 keeps the cast and the
    /// caller is expected to keep schematic size sane.
    pub fn volume_blocks(&self) -> u32 {
        let s = bounds_size(self.bounds);
        (s.x as u32) * (s.y as u32) * (s.z as u32)
    }

    /// Resolve a palette index to its numeric block id using the
    /// supplied registry. Returns AIR if the name doesn't exist in the
    /// registry (unknown schematics still paste — they just can't
    /// recreate the exact block).
    pub fn resolve(&self, palette_idx: u16, registry: &BlockRegistry) -> BlockId {
        self.palette
            .get(palette_idx as usize)
            .and_then(|name| registry.id_of(name))
            .unwrap_or(BlockId::AIR)
    }

    /// Save to `dest`. The file extension is not enforced; callers can
    /// use `.schem`, `.bin`, or whatever their tooling expects.
    pub fn save(&self, dest: &Path) -> io::Result<()> {
        let mut bytes = Vec::new();
        // Header (uncompressed).
        bytes.extend_from_slice(SCHEMATIC_MAGIC);
        bytes.extend_from_slice(&SCHEMATIC_FORMAT_VERSION.to_le_bytes());

        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_range(self.bounds);
        let ox = self.bounds.min.x.floor() as i32;
        let oy = self.bounds.min.y.floor() as i32;
        let oz = self.bounds.min.z.floor() as i32;
        let sx = (max_x - min_x).max(0) as u32;
        let sy = (max_y - min_y).max(0) as u32;
        let sz = (max_z - min_z).max(0) as u32;
        bytes.extend_from_slice(&ox.to_le_bytes());
        bytes.extend_from_slice(&oy.to_le_bytes());
        bytes.extend_from_slice(&oz.to_le_bytes());
        bytes.extend_from_slice(&sx.to_le_bytes());
        bytes.extend_from_slice(&sy.to_le_bytes());
        bytes.extend_from_slice(&sz.to_le_bytes());

        // Compressed body.
        let mut body = Vec::new();
        // Palette.
        body.extend_from_slice(&(self.palette.len() as u16).to_le_bytes());
        for name in &self.palette {
            let name_bytes = name.as_bytes();
            body.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            body.extend_from_slice(name_bytes);
        }
        // Voxels (already in BTreeMap-sorted order).
        body.extend_from_slice(&(self.voxels.len() as u32).to_le_bytes());
        for ((px, py, pz), pal_idx) in &self.voxels {
            body.extend_from_slice(&(*px as u16).to_le_bytes());
            body.extend_from_slice(&(*py as u16).to_le_bytes());
            body.extend_from_slice(&(*pz as u16).to_le_bytes());
            body.extend_from_slice(&pal_idx.to_le_bytes());
        }

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&body)?;
        let compressed = encoder.finish()?;
        bytes.extend_from_slice(&compressed);

        let file = File::create(dest)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes)?;
        writer.flush()?;
        Ok(())
    }

    /// Load from `src`. Header signature mismatches and unsupported
    /// versions return `InvalidData`. Decompression errors propagate as
    /// `io::Error` of their underlying kind.
    pub fn load(src: &Path, id: SchematicId) -> io::Result<Self> {
        let file = File::open(src)?;
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 9 + 4];
        reader.read_exact(&mut header)?;
        if header[..9] != b"SCHEMATIC"[..] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "schematic magic mismatch",
            ));
        }
        let version = u32::from_le_bytes([header[9], header[10], header[11], header[12]]);
        if version != SCHEMATIC_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported schematic format version {version}"),
            ));
        }
        let mut dims = [0u8; 24];
        reader.read_exact(&mut dims)?;
        let ox = i32::from_le_bytes(dims[0..4].try_into().unwrap());
        let oy = i32::from_le_bytes(dims[4..8].try_into().unwrap());
        let oz = i32::from_le_bytes(dims[8..12].try_into().unwrap());
        let sx = u32::from_le_bytes(dims[12..16].try_into().unwrap());
        let sy = u32::from_le_bytes(dims[16..20].try_into().unwrap());
        let sz = u32::from_le_bytes(dims[20..24].try_into().unwrap());

        let mut decoder = ZlibDecoder::new(reader);
        let mut body = Vec::new();
        decoder.read_to_end(&mut body)?;

        let mut r = BodyReader::new(&body);

        let palette_len = r.read_u16()? as usize;
        let mut palette = Vec::with_capacity(palette_len);
        for _ in 0..palette_len {
            let nlen = r.read_u16()? as usize;
            let mut buf = vec![0u8; nlen];
            r.read_exact_bytes(&mut buf)?;
            palette.push(
                String::from_utf8(buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?,
            );
        }

        let voxel_count = r.read_u32()? as usize;
        let mut voxels = BTreeMap::new();
        for _ in 0..voxel_count {
            let rx = r.read_u16()? as i32;
            let ry = r.read_u16()? as i32;
            let rz = r.read_u16()? as i32;
            let pal = r.read_u16()?;
            voxels.insert((rx, ry, rz), pal);
        }

        let bounds = Aabb {
            min: Vec3f::new(ox as f32, oy as f32, oz as f32),
            max: Vec3f::new(
                (ox + sx as i32) as f32,
                (oy + sy as i32) as f32,
                (oz + sz as i32) as f32,
            ),
        };

        Ok(Schematic {
            id,
            bounds,
            palette,
            voxels,
        })
    }

    /// Paste this schematic into `world` with `origin` as the
    /// bottom-south-west corner *after* rotation/mirror.
    /// Returns the number of distinct blocks written (not voxels
    /// emitted — unloaded chunks and old == new skips are not counted).
    pub fn paste<F: FnMut(BlockChange)>(
        &self,
        world: &World,
        origin: IVec3,
        rotation: Rotation90,
        mirror: MirrorAxes,
        mut record: F,
    ) -> usize {
        let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_range(self.bounds);
        let size = IVec3::new(max_x - min_x, max_y - min_y, max_z - min_z);
        let size_minus_one = IVec3::new(size.x - 1, size.y - 1, size.z - 1);
        let registry = world.registry_ref();
        let mut n = 0;
        for (tuple_pos, pal_idx) in &self.voxels {
            let rel = pos_to_iv3(*tuple_pos);
            let rotated = rotation.apply(rel);
            let mirrored = mirror.apply(rotated, size_minus_one);
            // Rotate + mirror operate in *capture-local* coords. The
            // shape still fits within the same `size` box (mirrored across
            // the centerline). Anchor the result by adding origin.
            let target = IVec3::new(
                origin.x + mirrored.x,
                origin.y + mirrored.y,
                origin.z + mirrored.z,
            );
            let block_id = self.resolve(*pal_idx, registry);
            if block_id.is_air() {
                continue;
            }
            n += paste_one(world, target, block_id, &mut record);
        }
        n
    }
}

impl World {
    /// Paste a `Schematic` into this `World` and register a
    /// [`SchematicEntity`] so it can be listed / queried later.
    ///
    /// Each *distinct* block change is forwarded to the `record`
    /// closure as a [`BlockChange`] before the entity registry is
    /// updated. This single-pass shape is what the Feature 4 dispatcher
    /// relies on: callers that want to batch the paste as a single undo
    /// entry funnel `record` through
    /// [`voxel_game::UndoRedoState::push_edit_batched`]. A second
    /// `schem.paste(…)` call alongside this method would double-write the
    /// world because [`Schematic::paste`] itself writes each block, so
    /// any caller wanting "paste + record" must use this method
    /// exclusively.
    ///
    /// Returns the post-paste entity (id + origin + rotation + mirror +
    /// effective bounds + count of distinct block changes), or `None` if
    /// zero blocks were written (unloaded chunks or `old == new`).
    pub fn paste_schematic<F: FnMut(BlockChange)>(
        &self,
        schem: &Schematic,
        origin: IVec3,
        rotation: Rotation90,
        mirror: MirrorAxes,
        mut record: F,
    ) -> Option<SchematicEntity> {
        let mut count = 0usize;
        let _ = schem.paste(self, origin, rotation, mirror, |change| {
            record(change);
            count += 1;
        });
        if count == 0 {
            return None;
        }
        let entity = SchematicEntity {
            id: schem.id.clone(),
            origin,
            rotation,
            mirror,
            post_bounds: rotated_bounds(schem.bounds, origin, rotation, mirror),
            pasted_blocks: count,
        };
        self.register_schematic_entity(entity.clone());
        Some(entity)
    }

    /// Register a `SchematicEntity` against this world. `pub(crate)`
    /// because external callers should go through
    /// [`Self::paste_schematic`] so `pasted_blocks` and `post_bounds`
    /// stay consistent with what was actually written.
    pub(crate) fn register_schematic_entity(&self, entity: SchematicEntity) {
        let mut map = self.schematic_entities.write();
        map.insert(entity.id.clone(), entity);
    }

    /// Snapshot of every schematic pasted into this world.
    pub fn pasted_schematics(&self) -> Vec<SchematicEntity> {
        let map = self.schematic_entities.read();
        let mut out: Vec<SchematicEntity> = map.values().cloned().collect();
        out.sort_by(|a, b| a.id.0.cmp(&b.id.0));
        out
    }

    /// Remove a pasted schematic's entity record (does not undo the
    /// pasted blocks).
    pub fn forget_schematic(&self, id: &SchematicId) -> bool {
        self.schematic_entities.write().remove(id).is_some()
    }
}

/// Lightweight tracker for a single paste operation. Doesn't correspond
/// to a live in-world entity — just metadata stored in the `World`'s
/// schematic registry.
///
/// `PartialEq` (not `Eq`) because `Aabb` only implements `PartialEq`
/// (it contains `f32`).
#[derive(Clone, Debug, PartialEq)]
pub struct SchematicEntity {
    pub id: SchematicId,
    pub origin: IVec3,
    pub rotation: Rotation90,
    pub mirror: MirrorAxes,
    /// Effective bounding box (in world coordinates) of the pasted
    /// region after rotation + mirror.
    pub post_bounds: Aabb,
    /// Number of distinct block changes that were actually written
    /// (after `old == new` and unloaded-chunk skips).
    pub pasted_blocks: usize,
}

// --- Internal helpers --------------------------------------------------

/// Wrapper around `&[u8]` that advances a cursor as it reads. Used by
/// [`Schematic::load`] to parse the deflated body.
struct BodyReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BodyReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        self.read_exact_bytes(&mut [0u8; 2])?;
        Ok(u16::from_le_bytes([
            self.bytes[self.pos - 2],
            self.bytes[self.pos - 1],
        ]))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        self.read_exact_bytes(&mut [0u8; 4])?;
        Ok(u32::from_le_bytes([
            self.bytes[self.pos - 4],
            self.bytes[self.pos - 3],
            self.bytes[self.pos - 2],
            self.bytes[self.pos - 1],
        ]))
    }

    fn read_exact_bytes(&mut self, dst: &mut [u8]) -> io::Result<()> {
        if self.pos + dst.len() > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "schematic body truncated",
            ));
        }
        dst.copy_from_slice(&self.bytes[self.pos..self.pos + dst.len()]);
        self.pos += dst.len();
        Ok(())
    }
}

fn aabb_range(bounds: Aabb) -> (i32, i32, i32, i32, i32, i32) {
    (
        bounds.min.x.floor() as i32,
        bounds.min.y.floor() as i32,
        bounds.min.z.floor() as i32,
        bounds.max.x.ceil() as i32,
        bounds.max.y.ceil() as i32,
        bounds.max.z.ceil() as i32,
    )
}

fn bounds_size(bounds: Aabb) -> IVec3 {
    let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_range(bounds);
    IVec3::new(max_x - min_x, max_y - min_y, max_z - min_z)
}

fn paste_one<F: FnMut(BlockChange)>(
    world: &World,
    target: IVec3,
    block: BlockId,
    record: &mut F,
) -> usize {
    let old = world.get_block(target.x, target.y, target.z);
    if old == block {
        return 0;
    }
    if world.set_block(target.x, target.y, target.z, block) {
        record(BlockChange {
            x: target.x,
            y: target.y,
            z: target.z,
            old,
            new: block,
        });
        1
    } else {
        0
    }
}

fn rotated_bounds(bounds: Aabb, origin: IVec3, rotation: Rotation90, mirror: MirrorAxes) -> Aabb {
    let (min_x, min_y, min_z, max_x, max_y, max_z) = aabb_range(bounds);
    let size = IVec3::new(max_x - min_x, max_y - min_y, max_z - min_z);
    let size_minus_one = IVec3::new(size.x - 1, size.y - 1, size.z - 1);
    let corners = [
        IVec3::new(0, 0, 0),
        IVec3::new(size.x - 1, 0, 0),
        IVec3::new(0, size.y - 1, 0),
        IVec3::new(size.x - 1, size.y - 1, 0),
        IVec3::new(0, 0, size.z - 1),
        IVec3::new(size.x - 1, 0, size.z - 1),
        IVec3::new(0, size.y - 1, size.z - 1),
        IVec3::new(size.x - 1, size.y - 1, size.z - 1),
    ];
    let mut xs = Vec::with_capacity(8);
    let mut ys = Vec::with_capacity(8);
    let mut zs = Vec::with_capacity(8);
    for c in corners {
        let r = rotation.apply(c);
        let m = mirror.apply(r, size_minus_one);
        xs.push(origin.x + m.x);
        ys.push(origin.y + m.y);
        zs.push(origin.z + m.z);
    }
    let min_x = *xs.iter().min().unwrap();
    let max_x = *xs.iter().max().unwrap() + 1;
    let min_y = *ys.iter().min().unwrap();
    let max_y = *ys.iter().max().unwrap() + 1;
    let min_z = *zs.iter().min().unwrap();
    let max_z = *zs.iter().max().unwrap() + 1;
    Aabb {
        min: Vec3f::new(min_x as f32, min_y as f32, min_z as f32),
        max: Vec3f::new(max_x as f32, max_y as f32, max_z as f32),
    }
}

// --- Tests -------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use std::sync::Arc;
    use voxel_core::ChunkPos;

    fn make_world() -> Arc<World> {
        let world = World::new(7);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));
        world
    }

    fn aabb(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) -> Aabb {
        Aabb {
            min: Vec3f::new(min_x, min_y, min_z),
            max: Vec3f::new(max_x, max_y, max_z),
        }
    }

    #[test]
    fn rotation_table_isometric_truth() {
        let p = IVec3::new(1, 0, 0);
        assert_eq!(Rotation90::Deg0.apply(p), IVec3::new(1, 0, 0));
        assert_eq!(Rotation90::Deg90.apply(p), IVec3::new(0, 0, 1));
        assert_eq!(Rotation90::Deg180.apply(p), IVec3::new(-1, 0, 0));
        assert_eq!(Rotation90::Deg270.apply(p), IVec3::new(0, 0, -1));
    }

    #[test]
    fn mirror_constants_and_contains() {
        assert!(MirrorAxes::ALL.contains(MirrorAxes::X));
        assert!(MirrorAxes::ALL.contains(MirrorAxes::Y));
        assert!(MirrorAxes::ALL.contains(MirrorAxes::Z));
        assert!(!MirrorAxes::X.contains(MirrorAxes::Y));
        assert!(!MirrorAxes::YZ.contains(MirrorAxes::X));
        assert!(MirrorAxes::XY.contains(MirrorAxes::X));
        assert!(MirrorAxes::XY.contains(MirrorAxes::Y));
    }

    #[test]
    fn mirror_apply_uses_size_not_origin() {
        let sz = IVec3::new(3, 3, 3);
        assert_eq!(
            MirrorAxes::X.apply(IVec3::new(0, 1, 2), sz),
            IVec3::new(3, 1, 2)
        );
        assert_eq!(
            MirrorAxes::X.apply(IVec3::new(3, 1, 2), sz),
            IVec3::new(0, 1, 2)
        );
        assert_eq!(
            MirrorAxes::NONE.apply(IVec3::new(0, 1, 2), sz),
            IVec3::new(0, 1, 2)
        );
    }

    #[test]
    fn schematic_id_equality_hash_and_display() {
        let a = SchematicId::new("castle");
        let b: SchematicId = "castle".into();
        let c = SchematicId::new("tree");
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = std::collections::HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert_eq!(format!("{}", a), "castle");
    }

    #[test]
    fn capture_roundtrip_preserves_blocks() {
        let world = make_world();
        let registry = BlockRegistry::with_builtins();
        world.set_block(0, 0, 0, BlockId(2));
        world.set_block(1, 0, 0, BlockId(2));
        world.set_block(2, 0, 0, BlockId(3));
        let schem = Schematic::capture(
            SchematicId::new("test"),
            aabb(-0.0, 0.0, 0.0, 3.0, 1.0, 1.0),
            &world,
            &registry,
        );
        assert_eq!(schem.voxel_count(), 3);
        let blocks: Vec<BlockId> = schem
            .voxels()
            .map(|(_, pi)| schem.resolve(pi, &registry))
            .collect();
        assert_eq!(blocks.iter().filter(|b| **b == BlockId(2)).count(), 2);
        assert_eq!(blocks.iter().filter(|b| **b == BlockId(3)).count(), 1);
        assert_eq!(schem.volume_blocks(), 3);
    }

    #[test]
    fn save_load_round_tripping() {
        let world = make_world();
        let registry = BlockRegistry::with_builtins();
        world.set_block(0, 0, 0, BlockId(2));
        world.set_block(2, 0, 1, BlockId(3));
        let schem = Schematic::capture(
            SchematicId::new("save_load"),
            aabb(0.0, 0.0, 0.0, 3.0, 1.0, 2.0),
            &world,
            &registry,
        );
        let dir = std::env::temp_dir().join("voxel_schematic_save_load_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("save_load.schem");
        let _ = std::fs::remove_file(&path);
        schem.save(&path).unwrap();
        let loaded = Schematic::load(&path, SchematicId::new("save_load")).unwrap();
        assert_eq!(loaded.voxel_count(), schem.voxel_count());
        assert_eq!(loaded.palette, schem.palette);
        let orig: Vec<(IVec3, u16)> = schem.voxels().collect();
        let back: Vec<(IVec3, u16)> = loaded.voxels().collect();
        assert_eq!(orig, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn paste_origin_identity_no_rotate_no_mirror() {
        let world = make_world();
        let registry = BlockRegistry::with_builtins();
        let schem = Schematic::capture(
            SchematicId::new("simple_aabb"),
            aabb(2.0, 0.0, 4.0, 4.0, 1.0, 5.0),
            &world,
            &registry,
        );
        let n = schem.paste(
            &world,
            IVec3::new(0, 0, 0),
            Rotation90::Deg0,
            MirrorAxes::NONE,
            |_c| {},
        );
        assert_eq!(n, 0);
    }

    #[test]
    fn paste_rotation_90_swaps_xz() {
        let world = make_world();
        let _registry = BlockRegistry::with_builtins();
        // 2x1x1 schematic with a stone at local (1,0,0). Deg90 rotation
        // about the Y axis sends (1, 0, 0) → (0, 0, 1) — i.e. the
        // X-binding becomes a Z-binding, proving the rotation path.
        let mut palette: Vec<String> = vec!["air".to_string()];
        palette.push("stone".to_string());
        let mut voxels = BTreeMap::new();
        voxels.insert((1, 0, 0), 1);
        let schem = Schematic {
            id: SchematicId::new("rot_test"),
            bounds: aabb(0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
            palette,
            voxels,
        };
        let mut changes: Vec<BlockChange> = Vec::new();
        let n = schem.paste(
            &world,
            IVec3::new(0, 0, 0),
            Rotation90::Deg90,
            MirrorAxes::NONE,
            |c| changes.push(c),
        );
        assert_eq!(n, 1);
        assert_eq!(changes[0].x, 0);
        assert_eq!(changes[0].z, 1);
    }

    #[test]
    fn paste_mirror_x_flips_x() {
        let world = make_world();
        let _registry = BlockRegistry::with_builtins();
        let mut palette = vec!["air".to_string()];
        palette.push("stone".to_string());
        let mut voxels = BTreeMap::new();
        voxels.insert((3, 0, 0), 1);
        let schem = Schematic {
            id: SchematicId::new("mirror_test"),
            bounds: aabb(0.0, 0.0, 0.0, 4.0, 1.0, 1.0),
            palette,
            voxels,
        };
        let mut changes: Vec<BlockChange> = Vec::new();
        let n = schem.paste(
            &world,
            IVec3::new(0, 0, 0),
            Rotation90::Deg0,
            MirrorAxes::X,
            |c| changes.push(c),
        );
        assert_eq!(n, 1);
        assert_eq!(changes[0].x, 0);
        assert_eq!(changes[0].y, 0);
        assert_eq!(changes[0].z, 0);
    }

    #[test]
    fn paste_skips_unloaded_chunks() {
        let world = World::new(11);
        let mut palette = vec!["air".to_string()];
        palette.push("stone".to_string());
        let mut voxels = BTreeMap::new();
        voxels.insert((0, 0, 0), 1);
        let schem = Schematic {
            id: SchematicId::new("empty_world"),
            bounds: aabb(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            palette,
            voxels,
        };
        let mut count = 0;
        let n = schem.paste(
            &world,
            IVec3::new(0, 0, 0),
            Rotation90::Deg0,
            MirrorAxes::NONE,
            |_| count += 1,
        );
        assert_eq!(n, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn paste_schematic_registers_entity_and_persists() {
        let world = make_world();
        let registry = BlockRegistry::with_builtins();
        world.set_block(0, 0, 0, BlockId(2));
        let schem = Schematic::capture(
            SchematicId::new("persist"),
            aabb(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            &world,
            &registry,
        );
        let entity = world
            .paste_schematic(
                &schem,
                IVec3::new(5, 0, 5),
                Rotation90::Deg0,
                MirrorAxes::NONE,
                |_change| {},
            )
            .expect("at least one block should have written");
        assert_eq!(entity.pasted_blocks, 1);
        assert_eq!(entity.origin, IVec3::new(5, 0, 5));
        assert_eq!(entity.post_bounds.min, Vec3f::new(5.0, 0.0, 5.0));
        assert_eq!(entity.post_bounds.max, Vec3f::new(6.0, 1.0, 6.0));

        let list = world.pasted_schematics();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, SchematicId::new("persist"));
        assert!(world.forget_schematic(&SchematicId::new("persist")));
        assert!(world.pasted_schematics().is_empty());
    }

    #[test]
    fn paste_schematic_returns_none_when_no_chunks_loaded() {
        // No chunks loaded anywhere → every schematic voxel is in a
        // missing chunk → paste_schematic must return None and the
        // registry must stay empty (no half-pasted entity left behind).
        let world = World::new(13);
        let mut palette = vec!["air".to_string()];
        palette.push("stone".to_string());
        let mut voxels = BTreeMap::new();
        voxels.insert((0, 0, 0), 1);
        let schem = Schematic {
            id: SchematicId::new("no_chunks"),
            bounds: aabb(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            palette,
            voxels,
        };
        let result = world.paste_schematic(
            &schem,
            IVec3::new(4, 4, 4),
            Rotation90::Deg0,
            MirrorAxes::NONE,
            |_change| {},
        );
        assert!(result.is_none(), "no chunks loaded → must be None");
        assert!(
            world.pasted_schematics().is_empty(),
            "no entity should be registered on a zero-change paste"
        );
    }

    #[test]
    fn rotated_bounds_reflects_post_transform() {
        let bounds = aabb(0.0, 0.0, 0.0, 2.0, 1.0, 4.0);
        let rotated = rotated_bounds(
            bounds,
            IVec3::new(10, 20, 30),
            Rotation90::Deg90,
            MirrorAxes::NONE,
        );
        // size = (2, 1, 4); rotated corners after Deg90 in (-3..=1)×{0}×{0,1}.
        // x ∈ {10 - 3, ..., 10+1} = {7, 8, 9, 10, 11}; y = {20, 21};
        // z = {30, 31}.
        assert_eq!(rotated.min.x, 7.0);
        assert_eq!(rotated.max.x, 11.0);
        assert_eq!(rotated.min.y, 20.0);
        assert_eq!(rotated.max.y, 21.0);
        assert_eq!(rotated.min.z, 30.0);
        assert_eq!(rotated.max.z, 32.0);
    }
}
