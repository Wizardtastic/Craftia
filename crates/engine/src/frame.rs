//! Per-frame engine orchestration: drain the chunk streamer, run the
//! fixed-timestep ECS schedule, update camera + day/night sky, render,
//! handle auto/screenshot capture.

use std::time::Instant;

use crate::edit;
use crate::edit::terrain::TerrainOp;
use crate::{spawn_particles_break, spawn_particles_place, update_particles};
use voxel_core::Point;
use voxel_render::UiDrawData;
use voxel_world::ChunkStreamEvent;

impl crate::EngineApp {
    /// Append the upload entries for `bundle` into the supplied `out` Vec.
    /// Each upload owns its vertex/index bytes via `to_vec()` on a slice
    /// cast from the bundle's POD arrays; this replaces the previous
    /// scratch-then-clone pattern that did two full copies per chunk with
    /// a single allocation. Only called from `frame()` in this module.
    fn extend_uploads(
        out: &mut Vec<voxel_render::ChunkUpload>,
        pos: voxel_core::ChunkPos,
        bundle: &voxel_world::ChunkMeshBundle,
    ) {
        if !bundle.opaque.is_empty() {
            out.push(voxel_render::ChunkUpload {
                pos,
                pass: voxel_render::MeshPass::Opaque,
                vertices: bytemuck::cast_slice(&bundle.opaque.vertices).to_vec(),
                indices: bytemuck::cast_slice(&bundle.opaque.indices).to_vec(),
                index_count: bundle.opaque.indices.len() as u32,
            });
        }
        if !bundle.transparent.is_empty() {
            out.push(voxel_render::ChunkUpload {
                pos,
                pass: voxel_render::MeshPass::Transparent,
                vertices: bytemuck::cast_slice(&bundle.transparent.vertices).to_vec(),
                indices: bytemuck::cast_slice(&bundle.transparent.indices).to_vec(),
                index_count: bundle.transparent.indices.len() as u32,
            });
        }
    }

    /// Run one render frame: drain hot-reload + streamer events, upload meshes,
    /// run the fixed-timestep sim, then render + present.
    pub(crate) fn frame(&mut self) {
        // ---- Hot-reload drain (runs before any GPU work so that atlas /
        // shader / config reloads produce a result on the very next frame). --
        // Non-blocking: we want the frame loop to keep ticking when no event
        // arrives. The watcher thread is the producer; the engine owns the
        // receiver exclusively through `self.hot_reload`.
        if let Some(watcher) = self.hot_reload.as_mut() {
            // Drain up to N events per frame so a burst of saves doesn't
            // block the frame loop forever.
            const MAX_EVENTS_PER_FRAME: usize = 32;
            for _ in 0..MAX_EVENTS_PER_FRAME {
                let Some(event) = watcher.receiver.try_recv().ok() else {
                    break;
                };
                match event {
                    voxel_render::HotReloadEvent::ShaderChanged(name) => {
                        if let Some(r) = self.render.renderer.as_mut() {
                            match r.reload_shader(&name) {
                                Ok(()) => {
                                    self.gameplay
                                        .chat
                                        .push_message(format!("[shader] reloaded {name}"));
                                }
                                Err(e) => {
                                    log::warn!("reload_shader({name}): {e}");
                                    self.gameplay
                                        .chat
                                        .push_message(format!("[shader] {name}: {e}"));
                                }
                            }
                        }
                    }
                    voxel_render::HotReloadEvent::TextureAtlasChanged
                    | voxel_render::HotReloadEvent::TexturePackChanged => {
                        if let Some(r) = self.render.renderer.as_mut() {
                            match r.reload_atlas() {
                                Ok(()) => {
                                    // Sync pack info from renderer to engine UI manager.
                                    self.texture_pack_manager.loaded_packs = r
                                        .pack_infos()
                                        .iter()
                                        .map(|p| crate::TexturePackInfo {
                                            name: p.name.clone(),
                                            description: p.description.clone(),
                                            version: p.version.clone(),
                                            author: p.author.clone(),
                                            tile_count: p.tile_count,
                                            animation_count: p.animation_count,
                                            enabled: p.enabled,
                                        })
                                        .collect();
                                    self.gameplay.chat.push_message("[atlas] reloaded".into());
                                }
                                Err(e) => {
                                    log::warn!("reload_atlas: {e}");
                                    self.gameplay.chat.push_message(format!("[atlas] {e}"));
                                }
                            }
                        }
                    }
                    voxel_render::HotReloadEvent::ConfigChanged => {
                        // Re-read the on-disk config and reconcile into the
                        // live engine config + the chunk streamer. We split
                        // fields by what can actually hot-swap at runtime:
                        //
                        //   * live-applied: renderer fog/textures/shader_dir,
                        //     chunk streamer load_radius, player movement
                        //     values, keybinds (input system reads live).
                        //   * mirror-only + warn: unload_radius (worker holds
                        //     the initial StreamConfig in its closure),
                        //     world day_length, world seed (terrain regen
                        //     requires restart).
                        let path = self.config.config_path.clone();
                        let text = match std::fs::read_to_string(&path) {
                            Ok(t) => t,
                            Err(e) => {
                                log::warn!("config reload: read {path:?} failed: {e}");
                                continue;
                            }
                        };
                        let new_settings =
                            match toml::from_str::<crate::settings::GameSettings>(&text) {
                                Ok(s) => s,
                                Err(e) => {
                                    log::warn!("config reload: parse {path:?} failed: {e}");
                                    self.gameplay
                                        .chat
                                        .push_message("[config] parse error".to_string());
                                    continue;
                                }
                            };

                        // 1) Renderer-side hot-swap. Any
                        //    warnings the renderer logs (vsync / validation
                        //    are log-only) are silently absorbed. We skip the
                        //    whole swap when `RendererConfig` didn't change,
                        //    so noisy TOML saves (comment-only edits, format
                        //    changes) don't churn the GPU.
                        let new_rc = new_settings.to_renderer_config();
                        if new_rc != self.config.render {
                            if let Some(r) = self.render.renderer.as_mut() {
                                match r.reload_config(&new_rc) {
                                    Ok(()) => {
                                        // Mirror the renderer-eligible fields so
                                        // subsequent runtime decisions stay
                                        // aligned with what's actually on the GPU.
                                        self.config.render = new_rc.clone();
                                        self.gameplay
                                            .chat
                                            .push_message("[config] renderer reloaded".into());
                                    }
                                    Err(e) => {
                                        log::warn!("reload_config: {e}");
                                        self.gameplay.chat.push_message(format!("[config] {e}"));
                                    }
                                }
                            } else {
                                // No renderer yet (pre-resumed); still mirror so
                                // the config survives a later build.
                                self.config.render = new_rc.clone();
                            }
                        }

                        // 2) World section: gated on whole-struct inequality
                        //    via `WorldSettings`'s `PartialEq` derive. Inside
                        //    the gate we still take per-field actions:
                        //    load_radius hot-swaps via the streamer's
                        //    Cmd::LoadRadius channel; unload_radius /
                        //    day_length / seed need a worker or world-regen
                        //    restart. We capture old values from
                        //    `self.config.world` (canonical mirror) and
                        //    keep the older per-field mirrors in sync so
                        //    downstream code that reads them continues to
                        //    work. `self.config.seed` is intentionally NOT
                        //    mirrored — a worker holds the original startup
                        //    seed for terrain generation.
                        if new_settings.world != self.config.world {
                            let new_load = new_settings.world.load_radius.max(0) as u32;
                            let old_load = self.config.world.load_radius.max(0) as u32;
                            let old_unload = self.config.world.unload_radius;
                            let old_day_length = self.config.world.day_length;
                            let old_seed = self.config.world.seed;

                            self.config.world = new_settings.world.clone();
                            self.config.stream.load_radius = new_settings.world.load_radius;
                            self.config.stream.unload_radius = new_settings.world.unload_radius;
                            self.config.day_length = new_settings.world.day_length;

                            if new_load != old_load {
                                if let Some(streamer) = &self.world_state.streamer {
                                    streamer.set_load_radius(new_load);
                                    log::info!(
                                        "config reload: load_radius {old_load} -> {new_load}"
                                    );
                                    self.gameplay.chat.push_message(format!(
                                        "[config] load_radius -> {new_load}"
                                    ));
                                }
                            }
                            if new_settings.world.unload_radius != old_unload {
                                log::info!(
                                    "config reload: unload_radius mirrored ({old_unload} -> {}); worker holds initial value, restart to apply",
                                    new_settings.world.unload_radius
                                );
                                self.gameplay.chat.push_message(format!(
                                    "[config] unload_radius = {} (needs restart)",
                                    new_settings.world.unload_radius
                                ));
                            }
                            if (new_settings.world.day_length - old_day_length).abs() > f64::EPSILON
                            {
                                log::info!(
                                    "config reload: day_length mirrored; restart to rescale the in-flight day cycle"
                                );
                                self.gameplay
                                    .chat
                                    .push_message("[config] day_length needs restart".into());
                            }
                            if new_settings.world.seed != old_seed {
                                log::info!(
                                    "config reload: seed changed; restart to regenerate terrain"
                                );
                                self.gameplay
                                    .chat
                                    .push_message("[config] seed needs restart".into());
                            }
                        }

                        // 3) Mirror player + keybinds. The movement and
                        //    input systems read through `self.config` so the
                        //    next frame picks them up. We compare via the
                        //    PartialEq derives on `PlayerConfig` /
                        //    `KeybindSettings` and skip the mirror (and chat
                        //    message) when nothing actually changed.
                        let new_player = new_settings.to_player_config();
                        let player_changed = new_player != self.config.player;
                        if player_changed {
                            self.config.player = new_player;
                        }
                        let keybinds_changed = new_settings.keys != self.config.keybinds;
                        if keybinds_changed {
                            self.config.keybinds = new_settings.keys.clone();
                        }
                        if player_changed || keybinds_changed {
                            self.gameplay
                                .chat
                                .push_message("[config] player + keybinds mirrored".into());
                        }
                    }
                }
            }
        }

        // Process queued audio events.
        self.audio.process_events();

        // Drain streamer events and sync GPU buffers. The `uploads` Vec is
        // allocated per-frame because `Renderer::upload_chunks` takes
        // ownership of the outer Vec. Full Phase 4 #12 reuse (a persistent
        // outer Vec whose capacity survives across frames) requires
        // `upload_chunks` to accept `&mut Vec<ChunkUpload>` + drain; until
        // then we allocate locally and let the renderer consume it.
        // Wall-clock ms spent polling streamer events + uploading chunk
        // meshes this frame, reported in the telemetry dashboard.
        let mut chunk_upload_ms = 0.0f32;
        if let Some(streamer) = &self.world_state.streamer {
            let upload_t0 = Instant::now();
            let events = streamer.poll_events();
            let mut uploads = Vec::new();
            for ev in events {
                match ev {
                    ChunkStreamEvent::MeshReady { pos, bundle } => {
                        Self::extend_uploads(&mut uploads, pos, &bundle);
                    }
                    ChunkStreamEvent::Unloaded(pos) => {
                        if let Some(r) = self.render.renderer.as_mut() {
                            r.remove_chunk(pos);
                        }
                    }
                    ChunkStreamEvent::Generated(_) => {}
                    ChunkStreamEvent::GpuMeshReady { .. } => {
                        // GPU compute meshing is not wired into the legacy
                        // renderer path yet; ignore these events.
                    }
                }
            }
            if !uploads.is_empty() {
                if let Some(r) = self.render.renderer.as_mut() {
                    r.upload_chunks(uploads);
                }
            }
            chunk_upload_ms = upload_t0.elapsed().as_secs_f32() * 1000.0;
        }

        // Fixed-timestep simulation.
        let now = Instant::now();
        let frame_dt = now.duration_since(self.input.last_time).as_secs_f64();
        self.input.last_time = now;
        self.input.frame_time = frame_dt;
        self.gameplay.console.tick_cursor(frame_dt);

        // Accumulate play time while in Playing state.
        if self.gameplay.game_state == crate::GameState::Playing {
            self.gameplay.play_time_accumulator += frame_dt;
        }

        // Rotate panorama on title screen (~0.5 deg/frame ΓåÆ full rotation in ~90s).
        if matches!(
            self.gameplay.game_state,
            crate::GameState::TitleScreen | crate::GameState::WorldSelect
        ) {
            self.gameplay.panorama_rotation += 0.00873; // ~0.5 degrees in radians
        }

        // Wait for the player's chunk to load before running physics, so the
        // player doesn't fall through unloaded terrain. Once the first chunk
        // arrives, snap the player to the surface.
        if !self.input.spawned {
            let p = self
                .simulation
                .player_pos()
                .unwrap_or(self.gameplay.spawn_pos);
            if self.world_state.world.is_block_loaded(
                p.x.floor() as i32,
                p.y.floor() as i32,
                p.z.floor() as i32,
            ) {
                // Find the surface: scan down from current Y for the first solid.
                let mut surface_y = p.y.floor() as i32;
                for y in ((p.y.floor() as i32 - 20).max(1)..=p.y.floor() as i32 + 5).rev() {
                    if self
                        .world_state
                        .world
                        .is_solid(p.x.floor() as i32, y, p.z.floor() as i32)
                    {
                        surface_y = y;
                        break;
                    }
                }
                // Place the player standing on the surface.
                let new_pos =
                    glam::Vec3::new(p.x, surface_y as f32 + 1.0 + voxel_game::PLAYER_HALF.y, p.z);
                self.simulation.set_player_pos(new_pos);
                self.input.spawned = true;
                log::info!(
                    "spawn ready: player at ({:.1}, {:.1}, {:.1}), surface_y={}",
                    new_pos.x,
                    new_pos.y,
                    new_pos.z,
                    surface_y
                );
                // Auto-capture runs skip the title screen so verification
                // screenshots show actual gameplay (terrain, water) instead
                // of the main menu.
                if self.config.capture_after_frames.is_some()
                    && self.gameplay.game_state == crate::GameState::TitleScreen
                {
                    self.enter_playing();
                    // Optional camera override (--campos/--camrot): teleport
                    // the player, enable flying (no falling), and point the
                    // camera for a deterministic verification shot. Applied
                    // once, right after spawn, so chunks stream around the
                    // override position before the capture frame fires.
                    if let Some(p) = self.config.capture_cam_pos {
                        let pos = glam::Vec3::new(p[0], p[1], p[2]);
                        self.simulation.set_player_pos(pos);
                        self.simulation.set_player_flying(true);
                        if let Some((yaw, pitch)) = self.config.capture_cam_rot {
                            if let Some(mut cam) = self.simulation.player_camera() {
                                cam.pos = pos + glam::Vec3::new(0.0, voxel_game::EYE_HEIGHT, 0.0);
                                cam.yaw = yaw;
                                cam.pitch = pitch;
                                self.simulation.set_player_camera(cam);
                            }
                        }
                    }
                }
            }
            // Chunk not loaded yet: fall through and skip the entire
            // simulation block (rendering still happens). The fixed-step
            // accumulator lives on `Simulation` and is only updated when
            // we actually call `tick_fixed`, so the wall-clock gap is
            // simply discarded rather than unleashing a catch-up burst
            // once the chunk finally arrives.
        }

        // Consume clicks once per frame.
        let mut clicks = self.input.input.take_clicks();

        // Handle block picker clicks.
        if self.gameplay.block_picker_open && clicks.left {
            self.handle_block_picker_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Handle ECS inspector clicks (left = pin the row under cursor).
        if self.gameplay.ecs_inspector && clicks.left {
            self.handle_ecs_inspector_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Handle pause-menu clicks (only in PauseMenu state).
        if self.gameplay.game_state == crate::GameState::PauseMenu && clicks.left {
            self.handle_pause_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Handle death screen clicks (when playing and player is dead).
        if self.gameplay.game_state == crate::GameState::Playing && clicks.left {
            let is_dead = self
                .simulation
                .ecs_world()
                .resource::<voxel_game::PlayerEntity>()
                .and_then(|p| p.0)
                .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Health>(e))
                .map(|h| h.dead)
                .unwrap_or(false);
            if is_dead {
                self.handle_death_screen_click();
                clicks.left = false;
                clicks.right = false;
            }
        }

        // Handle title screen clicks.
        if self.gameplay.game_state == crate::GameState::TitleScreen && clicks.left {
            self.handle_title_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Handle world select clicks.
        if self.gameplay.game_state == crate::GameState::WorldSelect && clicks.left {
            self.handle_world_select_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Handle settings menu clicks.
        if self.gameplay.game_state == crate::GameState::SettingsMenu && clicks.left {
            self.handle_settings_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Update slider dragging each frame while mouse is held.
        if self.gameplay.game_state == crate::GameState::SettingsMenu {
            self.update_settings_slider_drag();
        }

        // Run simulation only when playing AND the spawn chunk has
        // loaded. `Simulation::tick_fixed` returns the union of chunks
        // whose blocks changed during the water tick; we hand those to
        // the streamer for remeshing.
        let water_affected =
            if self.gameplay.game_state == crate::GameState::Playing && self.input.spawned {
                // Check if player is dead — if so, skip input and unlock cursor.
                let is_dead = self
                    .simulation
                    .ecs_world()
                    .resource::<voxel_game::PlayerEntity>()
                    .and_then(|p| p.0)
                    .and_then(|e| self.simulation.ecs_world().get::<voxel_game::Health>(e))
                    .map(|h| h.dead)
                    .unwrap_or(false);

                if is_dead {
                    // Unlock cursor on first detection of death.
                    if self.input.cursor_locked {
                        self.unlock_cursor();
                    }
                    // Don't pass any movement input while dead.
                    let snap = voxel_game::InputSnapshot::default();
                    self.simulation.set_player_input(snap);
                    // Still tick so systems like regen can run (but movement is zeroed).

                    self.simulation.tick_fixed(frame_dt)
                } else if self.gameplay.block_picker_open {
                    // Block picker (creative inventory) is open — no movement, no mouse look.
                    if self.input.cursor_locked {
                        self.unlock_cursor();
                    }
                    let snap = voxel_game::InputSnapshot::default();
                    self.simulation.set_player_input(snap);
                    self.simulation.tick_fixed(frame_dt)
                } else {
                    // Normal input flow.
                    let snap = voxel_game::InputSnapshot {
                        forward: self.input.input.held(voxel_game::input::Action::Forward),
                        back: self.input.input.held(voxel_game::input::Action::Back),
                        left: self.input.input.held(voxel_game::input::Action::Left),
                        right: self.input.input.held(voxel_game::input::Action::Right),
                        jump: self.input.input.held(voxel_game::input::Action::Jump),
                        sneak: self.input.input.held(voxel_game::input::Action::Sneak),
                        sprint: self.input.input.held(voxel_game::input::Action::Sprint),
                        flying: self.simulation.player_flying(),
                        mouse_delta: self.input.input.mouse_delta,
                        mining: self.input.input.held(voxel_game::input::Action::Attack),
                        use_item: self.input.input.clicks.right,
                    };
                    self.input.input.mouse_delta = (0.0, 0.0);
                    self.simulation.set_player_input(snap);
                    let affected = self.simulation.tick_fixed(frame_dt);
                    if self.simulation.timings_enabled() {
                        self.profiler.system_timings = self.simulation.last_frame_timings();
                    }
                    affected
                }
            } else {
                std::collections::HashSet::new()
            };
        if !water_affected.is_empty() {
            if let Some(streamer) = &self.world_state.streamer {
                for cp in water_affected {
                    streamer.request_remesh(cp);
                }
            }
        }

        // Wall-clock ms spent in `tick_water` across this frame's fixed steps.
        let water_tick_ms = self.simulation.take_water_tick_ms();

        // Refresh the cached camera resource from the player's transform +
        // current eye offset. This is the camera the renderer will use.
        let player_camera_input = self.simulation.player_transform_state();
        if let (Some((t, s)), Some(cam_res)) = (
            player_camera_input,
            self.simulation
                .ecs_world_mut()
                .resource_mut::<voxel_game::CameraResource>(),
        ) {
            voxel_game::update_camera_from_transform(&mut cam_res.0, &t, &s);
        }

        // Pin-follow override: if `gameplay.pinned_entity` is set,
        // overwrite the camera from that entity's Transform. Sit *after*
        // the player-block above so when `pinned_entity = None` the
        // existing camera behavior is identical. Pinned debug entities
        // don't carry a `PlayerState` component, so we synthesize one
        // with the canonical `EYE_HEIGHT` to keep the camera sitting at
        // a sensible offset above the entity's feet.
        if let Some(pinned) = self.gameplay.pinned_entity {
            let ecs = self.simulation.ecs_world();
            if ecs.is_alive(pinned) {
                if let Some(t_ref) = ecs.get::<voxel_game::Transform>(pinned) {
                    let t = *t_ref;
                    if let Some(cam_res) = self
                        .simulation
                        .ecs_world_mut()
                        .resource_mut::<voxel_game::CameraResource>()
                    {
                        let synth_state = voxel_game::PlayerState {
                            eye_offset: voxel_game::EYE_HEIGHT,
                            ..Default::default()
                        };
                        voxel_game::update_camera_from_transform(&mut cam_res.0, &t, &synth_state);
                    }
                }
            } else {
                // Pinned entity despawned — clear the pin so the camera
                // doesn't get stuck locked to a freed entity handle.
                self.gameplay.pinned_entity = None;
                self.gameplay
                    .chat
                    .push_message("Pin cleared (entity gone)".into());
            }
        }

        // Update audio listener position from camera.
        if let Some(camera) = self.simulation.player_camera() {
            let fwd = camera.forward();
            self.audio.push_event(voxel_audio::AudioEvent::SetListener {
                pos: camera.pos.into(),
                forward: [fwd.x, fwd.y, fwd.z],
                up: [0.0, 1.0, 0.0],
            });
        }

        // Keep the streamer centred on the player (even before spawn, so the
        // spawn chunk loads ASAP).
        let player_pos_now = self
            .simulation
            .player_pos()
            .unwrap_or(self.gameplay.spawn_pos);
        let camera_now = self.simulation.player_camera().unwrap_or_default();
        if let Some(streamer) = &self.world_state.streamer {
            // Only send focus if player moved to a different chunk.
            let player_chunk =
                voxel_core::math::block_to_chunk(voxel_core::math::world_to_block(player_pos_now));
            let player_chunk_v = voxel_core::math::chunk_origin(player_chunk).as_vec3();
            if (player_chunk_v - self.input.last_focus_pos).length_squared() > 1.0 {
                streamer.set_focus(player_pos_now);
                self.input.last_focus_pos = player_chunk_v;
            }
            // Only send sun_dir if it changed.
            let dp = self.day_params();
            let sun_dir = glam::Vec3::new(
                dp.sun_angle.cos() * 0.3,
                dp.sun_altitude,
                dp.sun_angle.sin() * 0.3,
            )
            .normalize();
            if (sun_dir - self.input.last_sun_dir).length_squared() > 0.0001 {
                streamer.set_sun_dir(sun_dir);
                self.input.last_sun_dir = sun_dir;
            }
            // Send frustum for mesh-building culling.
            let mut cam = camera_now;
            let h = self.render.window_size.1 as f32;
            cam.aspect = if h > 0.0 {
                self.render.window_size.0 as f32 / h
            } else {
                1.0
            };
            let vp = cam.view_projection();
            let frustum = voxel_core::Frustum::from_view_projection(vp);
            streamer.set_frustum(frustum);
        }

        // --- Minimap update (rate-limited) ---
        if self.gameplay.game_state == crate::GameState::Playing {
            let pos = self.simulation.player_pos();
            if let Some(p) = pos {
                let px = p.x.floor() as i32;
                let pz = p.z.floor() as i32;
                self.gameplay.map.check_dirty(px, pz);

                // Always mark dirty if the framebuffer is empty (first load or
                // chunks weren't ready on the previous attempt).
                if self.gameplay.map.framebuffer.is_empty() {
                    self.gameplay.map.dirty = true;
                }

                if self.gameplay.map.dirty {
                    let elapsed = self.gameplay.map.last_update.elapsed();
                    if elapsed >= self.gameplay.map.update_interval {
                        let samples = self
                            .world_state
                            .world
                            .sample_columns_chunked((px, pz), self.gameplay.map.radius_blocks);
                        let sample_count = samples.len();
                        self.gameplay.map.samples = samples;
                        self.gameplay
                            .map
                            .rebuild_framebuffer(&self.world_state.world.registry());

                        // Only upload if we got data.
                        if sample_count > 0 {
                            if let Some(r) = self.render.renderer.as_mut() {
                                r.upload_minimap_texture(&self.gameplay.map.framebuffer);
                            }
                            log::debug!(
                                "minimap: uploaded {} samples, {} bytes",
                                sample_count,
                                self.gameplay.map.framebuffer.len()
                            );
                        } else {
                            // No samples yet — stay dirty so we retry next interval.
                            self.gameplay.map.dirty = true;
                            log::debug!("minimap: no samples at ({}, {}), retrying", px, pz);
                        }
                        self.gameplay.map.last_update = std::time::Instant::now();
                    }
                }
            }
        }

        // Block interactions only when playing, chat closed, picker closed, edit mode off.
        if (clicks.left || clicks.right)
            && self.input.cursor_locked
            && self.gameplay.game_state == crate::GameState::Playing
            && !self.gameplay.chat.open
            && !self.gameplay.block_picker_open
            && !self.gameplay.edit.mode.is_active()
        {
            if let Some(streamer) = &self.world_state.streamer {
                let eye = self.simulation.player_eye_pos().unwrap_or(camera_now.pos);
                let result = voxel_game::BlockAction::apply(
                    &self.world_state.world,
                    streamer,
                    &mut self.gameplay.hotbar,
                    eye,
                    camera_now.forward(),
                    clicks,
                    player_pos_now,
                );
                if !result.edits.is_empty() {
                    // Clone edits for particle spawning before moving into undo.
                    let edits_for_particles = result.edits.clone();
                    self.gameplay.undo_redo.push(voxel_game::EditAction {
                        edits: result.edits,
                    });
                    // Spawn particles on block break.
                    if result.broke {
                        if let Some(hit) = result.target {
                            // Skip if block ID is out of range (safety check).
                            if hit.block_id.0 as usize >= self.world_state.world.registry().count()
                            {
                                // Block ID invalid, skip particles.
                            } else {
                                let pos = glam::Vec3::new(
                                    hit.block.x as f32 + 0.5,
                                    hit.block.y as f32 + 0.5,
                                    hit.block.z as f32 + 0.5,
                                );
                                let reg = self.world_state.world.registry();
                                let def = reg.get(hit.block_id);
                                let color = def.map_color;
                                let normal = camera_now.forward();
                                spawn_particles_break(
                                    &mut self.render.renderer,
                                    pos,
                                    color,
                                    normal,
                                );
                            }
                        }
                    }
                    // Spawn particles on block place (mirrors the break branch
                    // above). Iterates `edits_for_particles` because brush-based
                    // placement may place multiple blocks per click.
                    if result.placed {
                        let reg = self.world_state.world.registry();
                        for edit in &edits_for_particles {
                            let block_id = voxel_core::BlockId(edit.new_block);
                            // Skip if block ID is out of range (safety check).
                            if block_id.0 as usize >= reg.count() {
                                continue;
                            }
                            let pos = glam::Vec3::new(
                                edit.x as f32 + 0.5,
                                edit.y as f32 + 0.5,
                                edit.z as f32 + 0.5,
                            );
                            let color = reg.get(block_id).map_color;
                            spawn_particles_place(&mut self.render.renderer, pos, color);
                        }
                    }
                    // Trigger mining swing animation on PlayerState.
                    if clicks.left {
                        if let Some(player_entity) = self.simulation.player_entity() {
                            if let Some(state) = self
                                .simulation
                                .ecs_world_mut()
                                .get_mut::<voxel_game::PlayerState>(player_entity)
                            {
                                state.mining_swing = 1.0;
                            }
                        }
                    }
                }
            }
        }

        // Update looked_at_block from raycast (for outline rendering and mining).
        if self.gameplay.game_state == crate::GameState::Playing && self.input.cursor_locked {
            let eye = self.simulation.player_eye_pos().unwrap_or(camera_now.pos);
            let ray = voxel_core::Ray::new(eye, camera_now.forward(), 6.0);
            if let Some(hit) = voxel_physics::raycast_voxels(&self.world_state.world, ray) {
                self.gameplay.looked_at_block = Some([hit.block.x, hit.block.y, hit.block.z]);
                // Update ECS resource for mining system.
                if let Some(look) = self
                    .simulation
                    .ecs_world_mut()
                    .resource_mut::<voxel_game::PlayerLookTarget>()
                {
                    look.block = Some([hit.block.x, hit.block.y, hit.block.z]);
                    look.block_id =
                        self.world_state
                            .world
                            .get_block(hit.block.x, hit.block.y, hit.block.z);
                }
                // Update crosshair mode based on what we're looking at.
                let block_id =
                    self.world_state
                        .world
                        .get_block(hit.block.x, hit.block.y, hit.block.z);
                let reg = self.world_state.world.registry();
                let def = reg.get(block_id);
                if def.name.as_ref() == "chest" || def.name.as_ref() == "crafting_table" {
                    self.gameplay.crosshair_mode = crate::CrosshairMode::Interact;
                } else {
                    self.gameplay.crosshair_mode = crate::CrosshairMode::BlockTarget;
                }
            } else {
                self.gameplay.looked_at_block = None;
                if let Some(look) = self
                    .simulation
                    .ecs_world_mut()
                    .resource_mut::<voxel_game::PlayerLookTarget>()
                {
                    look.block = None;
                    look.block_id = voxel_core::BlockId::AIR;
                }
                self.gameplay.crosshair_mode = crate::CrosshairMode::Default;
            }
        } else {
            self.gameplay.looked_at_block = None;
            if let Some(look) = self
                .simulation
                .ecs_world_mut()
                .resource_mut::<voxel_game::PlayerLookTarget>()
            {
                look.block = None;
                look.block_id = voxel_core::BlockId::AIR;
            }
            self.gameplay.crosshair_mode = crate::CrosshairMode::Default;
        }

        // Update mining progress for crack overlay.
        if self.gameplay.game_state == crate::GameState::Playing {
            let player_entity = self.simulation.player_entity();
            if let Some(entity) = player_entity {
                let mining = self
                    .simulation
                    .ecs_world()
                    .get::<voxel_game::MiningProgress>(entity);
                if let Some(mining) = mining {
                    if let Some(target) = mining.target_block {
                        self.gameplay.mining_crack = Some((target, mining.crack_stage));
                    } else {
                        self.gameplay.mining_crack = None;
                    }
                } else {
                    self.gameplay.mining_crack = None;
                }
            }
        }

        // Update particles (CPU sim + GPU upload). Owned by the renderer;
        // the engine no longer hand-rolls a CPU-projected UI overlay (that
        // path was removed in Phase 1 in favour of real 3D billboards).
        update_particles(&mut self.render.renderer, frame_dt as f32);

        // Advance game time (day/night cycle). Modulo so a long frame wraps
        // correctly even if frame_dt > day_length.
        if self.gameplay.game_state == crate::GameState::Playing {
            self.gameplay.game_time =
                (self.gameplay.game_time + frame_dt) % self.gameplay.day_length;
        }

        // Sync game time to ECS resource so systems can read it.
        if let Some(gt) = self
            .simulation
            .ecs_world_mut()
            .resource_mut::<voxel_game::GameTimeResource>()
        {
            gt.0 = self.gameplay.game_time;
        }

        // Sync hotbar selection to ECS resource so mining system can read it.
        if let Some(hotbar_res) = self
            .simulation
            .ecs_world_mut()
            .resource_mut::<voxel_game::HotbarResource>()
        {
            let selected = self
                .gameplay
                .hotbar
                .selected_block()
                .unwrap_or(voxel_core::BlockId::AIR);
            hotbar_res.selected_block = selected;
            // Keep the selected tool metadata synchronized with the slot. The
            // current default palette carries tier 0, while tool-aware callers
            // can attach a higher tier to a selected slot explicitly.
            hotbar_res.selected_tool_tier = self.gameplay.hotbar.selected_tool_tier();
            // Look up the tile index for the held item rendering.
            if !selected.is_air() {
                let reg = self.world_state.world.registry();
                let def = reg.get(selected);
                // Use the top face (PosY = index 3) tile for the held item.
                hotbar_res.tile = def.textures.tiles[3] as u32;
            } else {
                hotbar_res.tile = 0;
            }
        }

        // Drain one queued console script command per frame (from /exec).
        if let Some(script) = &mut self.gameplay.console_script {
            if let Some(line) = script.first().cloned() {
                script.remove(0);
                if script.is_empty() {
                    self.gameplay.console_script = None;
                }
                if line.starts_with('/') {
                    let pos = self
                        .simulation
                        .player_pos()
                        .unwrap_or(self.gameplay.spawn_pos);
                    let result = voxel_game::ChatState::parse_command(&line, pos);
                    match &result {
                        voxel_game::CommandResult::EcsList
                        | voxel_game::CommandResult::EcsInspect { .. }
                        | voxel_game::CommandResult::EcsResources
                        | voxel_game::CommandResult::EcsResource { .. }
                        | voxel_game::CommandResult::Get { .. }
                        | voxel_game::CommandResult::Set { .. }
                        | voxel_game::CommandResult::Exec(_) => {
                            let output = self.execute_console_command(result);
                            for line in output {
                                self.gameplay.console.println(line);
                            }
                        }
                        _ => {
                            self.execute_command(result);
                        }
                    }
                }
            }
        }

        // --- World editing brush / selection ---
        if self.gameplay.edit.mode.is_active() {
            if let Some(camera) = self.simulation.player_camera() {
                let ray = voxel_core::Ray::new(camera.pos, camera.forward(), 100.0);
                if let Some(hit) = voxel_physics::raycast_voxels(&self.world_state.world, ray) {
                    let center = hit.block + hit.normal;
                    self.gameplay.edit.brush_center = Some(center);
                    self.gameplay.edit.preview_valid = true;

                    // Check if click is outside UI panels. The editor draws
                    // in logical pixels now, so compare against the logical
                    // window size to match the panel rects.
                    let Point { x: mx, y: my } = self.gameplay.mouse_pos;
                    let (sw, sh) = self.render.logical_size();
                    let in_menu_bar = my < edit::theme::MENU_BAR_H;
                    let in_status_bar = my > sh - edit::theme::STATUS_BAR_H;
                    let in_cat_bar = mx < edit::theme::CAT_BAR_W;
                    let in_left_panel = (edit::theme::CAT_BAR_W
                        ..edit::theme::CAT_BAR_W + edit::theme::LEFT_PANEL_W)
                        .contains(&mx)
                        && my >= edit::theme::MENU_BAR_H
                        && my < sh - edit::theme::STATUS_BAR_H;
                    let in_right_panel = mx > sw - edit::theme::RIGHT_PANEL_W
                        && my >= edit::theme::MENU_BAR_H
                        && my < sh - edit::theme::STATUS_BAR_H;
                    let click_in_ui = in_menu_bar
                        || in_status_bar
                        || in_cat_bar
                        || in_left_panel
                        || in_right_panel;

                    // --- Selection tool interaction ---
                    if let Some(sel) = self.gameplay.edit.select_mut() {
                        // Left click starts/updates drag.
                        if clicks.left && (self.input.cursor_locked || !click_in_ui) {
                            if !sel.dragging {
                                sel.start_drag(hit.block);
                            }
                            clicks.left = false;
                        }
                        // Update drag while dragging (each frame).
                        if sel.dragging {
                            sel.update_drag(hit.block);
                        }
                        // Right click clears selection.
                        if clicks.right && (self.input.cursor_locked || !click_in_ui) {
                            sel.clear();
                            clicks.right = false;
                        }
                    }

                    // --- Brush tool interaction ---
                    if self.gameplay.edit.brush_ref().is_some() {
                        // Apply brush on left click.
                        if clicks.left && (self.input.cursor_locked || !click_in_ui) {
                            let affected = edit::brush::apply_brush(
                                &mut self.gameplay.edit,
                                &self.world_state.world,
                                center,
                                &mut self.gameplay.undo_redo,
                            );
                            if let Some(streamer) = &self.world_state.streamer {
                                for cp in affected {
                                    streamer.request_remesh(cp);
                                }
                            }
                            clicks.left = false;
                        }

                        // Pick block on right click. Shift+right-click picks replace target.
                        if clicks.right && (self.input.cursor_locked || !click_in_ui) {
                            let shift_held =
                                self.input.input.held(voxel_game::input::Action::Sneak);
                            if shift_held
                                && self.gameplay.edit.brush_ref().is_some_and(|b| b.replace)
                            {
                                edit::brush::pick_replace_target(
                                    &mut self.gameplay.edit,
                                    &self.world_state.world,
                                    hit.block,
                                );
                            } else {
                                edit::brush::pick_brush_block(
                                    &mut self.gameplay.edit,
                                    &self.world_state.world,
                                    hit.block,
                                );
                            }
                            clicks.right = false;
                        }
                    }

                    // --- Terrain tool interaction ---
                    if self.gameplay.edit.terrain_ref().is_some()
                        && clicks.left
                        && (self.input.cursor_locked || !click_in_ui)
                    {
                        let terrain = self.gameplay.edit.terrain_ref().unwrap().clone();
                        let affected = match &terrain.op {
                            TerrainOp::Raise { amount } => edit::terrain::apply_raise(
                                &self.world_state.world,
                                center,
                                terrain.radius,
                                *amount as i32,
                                terrain.block,
                                &mut self.gameplay.undo_redo,
                            ),
                            TerrainOp::Lower { amount } => edit::terrain::apply_lower(
                                &self.world_state.world,
                                center,
                                terrain.radius,
                                *amount as i32,
                                &mut self.gameplay.undo_redo,
                            ),
                            TerrainOp::Flatten { target_height } => {
                                let ty = target_height.unwrap_or(center.y);
                                edit::terrain::apply_flatten(
                                    &self.world_state.world,
                                    center,
                                    terrain.radius,
                                    ty,
                                    terrain.block,
                                    &mut self.gameplay.undo_redo,
                                )
                            }
                            TerrainOp::Smooth { iterations } => edit::terrain::apply_smooth(
                                &self.world_state.world,
                                center,
                                terrain.radius,
                                *iterations,
                                terrain.block,
                                &mut self.gameplay.undo_redo,
                            ),
                            TerrainOp::Noise {
                                scale,
                                amplitude,
                                seed,
                            } => edit::terrain::apply_noise(
                                &self.world_state.world,
                                center,
                                edit::terrain::NoiseParams {
                                    radius: terrain.radius,
                                    scale: *scale,
                                    amplitude: *amplitude,
                                    seed: *seed,
                                },
                                terrain.block,
                                &mut self.gameplay.undo_redo,
                            ),
                        };
                        if let Some(streamer) = &self.world_state.streamer {
                            for cp in affected {
                                streamer.request_remesh(cp);
                            }
                        }
                        clicks.left = false;
                    }

                    // --- Paint tool interaction ---
                    if self.gameplay.edit.paint_ref().is_some()
                        && clicks.left
                        && (self.input.cursor_locked || !click_in_ui)
                    {
                        let paint = self.gameplay.edit.paint_ref().unwrap().clone();
                        let affected = edit::paint::apply_gradient(
                            &paint,
                            &self.world_state.world,
                            center,
                            &mut self.gameplay.undo_redo,
                        );
                        if let Some(streamer) = &self.world_state.streamer {
                            for cp in affected {
                                streamer.request_remesh(cp);
                            }
                        }
                        clicks.left = false;
                    }

                    // --- Filter tool interaction ---
                    // Filters apply via UI button, not click in world.
                } else {
                    self.gameplay.edit.brush_center = None;
                    self.gameplay.edit.preview_valid = false;
                }
            }

            // Finalize selection drag when mouse is released (not held).
            if let Some(sel) = self.gameplay.edit.select_mut() {
                if sel.dragging && !clicks.left && !self.input.input.clicks.left {
                    sel.end_drag();
                }
            }
        }

        // Build UI overlay.
        let ui: UiDrawData = self.build_ui();

        // Collect entity render data from ECS (entities with Mesh + Transform).
        // Sort opaque front-to-back, transparent back-to-front for correct blending.
        // Partition held items (ALWAYS depth) from world entities (LESS_OR_EQUAL depth).
        let player_pos = self
            .simulation
            .player_pos()
            .unwrap_or(self.gameplay.spawn_pos);
        let (entity_data, held_item_data): (Vec<_>, Vec<_>) = {
            let ecs = self.simulation.ecs_world();
            let mut opaque = Vec::new();
            let mut transparent = Vec::new();
            let mut held = Vec::new();
            for arch in ecs.archetypes() {
                let has_mesh = arch.has::<voxel_game::Mesh>();
                let has_transform = arch.has::<voxel_game::Transform>();
                if !has_mesh || !has_transform {
                    continue;
                }
                for &entity in arch.entities().iter() {
                    if let (Some(mesh), Some(transform)) = (
                        ecs.get::<voxel_game::Mesh>(entity),
                        ecs.get::<voxel_game::Transform>(entity),
                    ) {
                        let data = voxel_render::entity::EntityRenderData {
                            pos: transform.pos,
                            rot: transform.rot,
                            tile: mesh.tile,
                            billboard: mesh.billboard,
                            half_size: mesh.half_size,
                            transparent: mesh.transparent,
                            held_item: false,
                        };
                        if mesh.transparent {
                            transparent.push(data);
                        } else {
                            opaque.push(data);
                        }
                    }
                }
            }

            // Sort opaque front-to-back (ascending squared distance).
            opaque.sort_by(|a, b| {
                let da = (a.pos - player_pos).length_squared();
                let db = (b.pos - player_pos).length_squared();
                da.total_cmp(&db)
            });
            // Sort transparent back-to-front (descending squared distance).
            transparent.sort_by(|a, b| {
                let da = (a.pos - player_pos).length_squared();
                let db = (b.pos - player_pos).length_squared();
                db.total_cmp(&da)
            });
            opaque.extend(transparent);

            // Check HeldBlock component on player for held item rendering.
            if let Some(player_entity) = self.simulation.player_entity() {
                if let Some(held_block) = self
                    .simulation
                    .ecs_world()
                    .get::<voxel_game::HeldBlock>(player_entity)
                {
                    if held_block.in_first_person && held_block.tile != 0 {
                        let cam = self.simulation.player_camera().unwrap_or_default();
                        let forward = cam.forward();
                        let right = cam.right();
                        let up = glam::Vec3::Y;

                        // Get player state for bob and swing.
                        let player_state = self
                            .simulation
                            .ecs_world()
                            .get::<voxel_game::PlayerState>(player_entity)
                            .copied();
                        let bob_phase = player_state.map(|s| s.bob_phase).unwrap_or(0.0);
                        let mining_swing = player_state.map(|s| s.mining_swing).unwrap_or(0.0);

                        // Compute bob offset (synced with camera bob, amplified 2x).
                        let bob_v = bob_phase.sin() * 0.025;
                        let bob_h = (bob_phase * 2.0).sin() * 0.008;
                        let bob_vec = glam::Vec3::new(bob_h, bob_v, 0.0);

                        // Compute swing rotation (ease-out arc).
                        let swing_rot = if mining_swing > 0.0 {
                            let progress = 1.0 - mining_swing;
                            let eased = 1.0 - (1.0 - progress).powi(2);
                            glam::Quat::from_rotation_z(-eased * 1.0)
                        } else {
                            glam::Quat::IDENTITY
                        };

                        // Position held item in bottom-right of view.
                        let base_offset = glam::Vec3::new(0.4, -0.3, -0.5);
                        let pos = cam.pos
                            + forward * base_offset.z
                            + right * base_offset.x
                            + up * base_offset.y
                            + bob_vec;
                        held.push(voxel_render::entity::EntityRenderData {
                            pos,
                            rot: swing_rot,
                            tile: held_block.tile,
                            billboard: false,
                            half_size: 0.15,
                            transparent: false,
                            held_item: true,
                        });

                        // Add arm below the held item.
                        let arm_offset = glam::Vec3::new(0.25, -0.38, -0.45);
                        let arm_pos = cam.pos
                            + forward * arm_offset.z
                            + right * arm_offset.x
                            + up * arm_offset.y
                            + bob_vec * 0.7; // Arm bobs slightly less
                                             // Arm uses a skin-colored tile (use tile 0 with a special color push).
                                             // For now, use a reserved tile index or the held block tile with different size.
                        held.push(voxel_render::entity::EntityRenderData {
                            pos: arm_pos,
                            rot: swing_rot,
                            tile: held_block.tile, // Will be replaced with arm texture tile
                            billboard: false,
                            half_size: 0.08, // Narrower than held item
                            transparent: false,
                            held_item: true,
                        });
                    }
                }
            }

            (opaque, held)
        };

        // Update sky parameters for day/night.
        let dp = self.day_params();
        let sun_dir = [
            dp.sun_angle.cos() * 0.3,
            dp.sun_altitude,
            dp.sun_angle.sin() * 0.3,
        ];
        self.world_state
            .world
            .set_sun_dir(glam::Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]).normalize());

        // Detect if camera is underwater for visual effects.
        let eye_block = voxel_core::math::world_to_block(camera_now.pos);
        let underwater = self
            .world_state
            .world
            .is_liquid(eye_block.x, eye_block.y, eye_block.z);

        // Build & push the per-tile material lookup table (chunk shader
        // binding 5). The registry + scalars are read once per frame; cost
        // is ~256 std430 entries of plain-struct writes, well under 0.01 ms
        // on any modern CPU. If profiling later shows it matters we can
        // hoist to a cache invalidated on registry/config edits.
        let world_registry = self.world_state.world.registry().clone();
        let material_table = Self::build_material_table_from_registry(
            &world_registry,
            self.config.water_y,
            self.config.wet_edge_strength,
            self.config.caustics_strength,
            self.config.leaves_sss_strength,
        );
        if let Some(r) = self.render.renderer.as_mut() {
            r.set_tile_material_table(material_table);
        }

        // Render.
        let mut camera = camera_now;
        let h = self.render.window_size.1 as f32;
        camera.aspect = if h > 0.0 {
            self.render.window_size.0 as f32 / h
        } else {
            1.0
        };
        if let Some(r) = self.render.renderer.as_mut() {
            r.set_sky(
                dp.horizon,
                dp.zenith,
                dp.fog,
                dp.daylight.max(0.15),
                underwater,
            );
            r.set_sun_dir(sun_dir);
            if self.config.shadow_enabled {
                let (cascade_vps, cascade_splits, light_dir_and_bias) = compute_shadow_cascades(
                    &camera,
                    glam::Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]),
                    0.1,
                    self.config.render.fog_distance,
                );
                r.set_shadow_data(cascade_vps, cascade_splits, light_dir_and_bias);
            }
            r.set_post_params(
                self.config.exposure,
                self.config.vignette_strength,
                self.gameplay.game_time as f32,
                underwater,
            );
            r.set_ssao_params(
                self.config.ssao_radius,
                self.config.ssao_bias,
                self.config.ssao_strength,
                self.config.ssao_enabled,
            );
            r.set_reflection_strength(self.config.reflection_strength);
            {
                let ext = r.extent();
                r.set_proj_params(camera.near, camera.far, ext.width as f32, ext.height as f32);
            }

            // Collect overlay data (brush wireframe preview + selection wireframe).
            let mut overlay_data =
                if let edit::EditModeState::Active { tool } = &self.gameplay.edit.mode {
                    if let Some(center) = self.gameplay.edit.brush_center {
                        let valid = self.gameplay.edit.preview_valid;
                        edit::brush::brush_wireframe(center, tool.shape(), tool.radius(), valid)
                    } else {
                        voxel_render::overlay::OverlayData::default()
                    }
                } else {
                    voxel_render::overlay::OverlayData::default()
                };
            // Add selection wireframe overlay if selection tool is active.
            if let Some(sel) = self.gameplay.edit.select_ref() {
                let sel_data = edit::select::selection_wireframe(sel);
                overlay_data.lines.extend(sel_data.lines);
            }

            // Add block selection outline (when not in edit mode).
            if !self.gameplay.edit.mode.is_active() {
                if let Some(block_pos) = self.gameplay.looked_at_block {
                    let x = block_pos[0] as f32;
                    let y = block_pos[1] as f32;
                    let z = block_pos[2] as f32;
                    let outline_color = [0, 0, 0, 255]; // Black outline
                    let _inner_color = [255, 255, 255, 64]; // White with low alpha

                    // 12 edges of a unit cube.
                    let edges = [
                        // Bottom face
                        ([x, y, z], [x + 1.0, y, z]),
                        ([x + 1.0, y, z], [x + 1.0, y, z + 1.0]),
                        ([x + 1.0, y, z + 1.0], [x, y, z + 1.0]),
                        ([x, y, z + 1.0], [x, y, z]),
                        // Top face
                        ([x, y + 1.0, z], [x + 1.0, y + 1.0, z]),
                        ([x + 1.0, y + 1.0, z], [x + 1.0, y + 1.0, z + 1.0]),
                        ([x + 1.0, y + 1.0, z + 1.0], [x, y + 1.0, z + 1.0]),
                        ([x, y + 1.0, z + 1.0], [x, y + 1.0, z]),
                        // Vertical edges
                        ([x, y, z], [x, y + 1.0, z]),
                        ([x + 1.0, y, z], [x + 1.0, y + 1.0, z]),
                        ([x + 1.0, y, z + 1.0], [x + 1.0, y + 1.0, z + 1.0]),
                        ([x, y, z + 1.0], [x, y + 1.0, z + 1.0]),
                    ];

                    for (a, b) in edges {
                        overlay_data.lines.push(voxel_render::overlay::OverlayLine {
                            a,
                            b,
                            color: outline_color,
                        });
                    }
                }
            }

            // Add mining crack overlay.
            if let Some((crack_pos, crack_stage)) = self.gameplay.mining_crack {
                if crack_stage > 0 {
                    let x = crack_pos[0] as f32;
                    let y = crack_pos[1] as f32;
                    let z = crack_pos[2] as f32;
                    // Draw crack lines based on stage (more lines = more cracks).
                    let crack_alpha = (crack_stage as f32 / 10.0 * 255.0) as u8;
                    let crack_color = [128, 128, 128, crack_alpha];

                    // Draw random crack lines based on stage.
                    let line_count = crack_stage as usize + 1;
                    for i in 0..line_count {
                        let t = i as f32 / line_count as f32;
                        let start = [x + t, y + 0.01, z + 0.01];
                        let end = [x + t * 0.5, y + 0.99, z + 0.99];
                        overlay_data.lines.push(voxel_render::overlay::OverlayLine {
                            a: start,
                            b: end,
                            color: crack_color,
                        });
                    }
                }
            }

            r.set_overlay(overlay_data);

            // Determine if we should show the panorama (title screen or world select).
            let show_panorama = matches!(
                self.gameplay.game_state,
                crate::GameState::TitleScreen | crate::GameState::WorldSelect
            );

            if let Err(e) = r.draw_frame(voxel_render::FrameInput {
                camera,
                ui: Some(&ui),
                game_time: self.gameplay.game_time as f32,
                underwater,
                world_entities: &entity_data,
                held_items: &held_item_data,
                show_panorama,
                panorama_rotation: self.gameplay.panorama_rotation,
            }) {
                log::error!("draw_frame: {e}");
            }
            // Collect profiler data.
            if self.profiler.enabled {
                self.profiler.end_frame(self.input.frame_time * 1000.0);
                let gpu = r.latest_timings();
                self.profiler.gpu_timings.push_back(gpu);
                if self.profiler.gpu_timings.len() > 120 {
                    self.profiler.gpu_timings.pop_front();
                }
            }

            // Collect telemetry (always, regardless of dashboard visibility).
            {
                let gpu = r.latest_timings();
                let (alloc_bytes, reserved_bytes) = r.allocator_stats();
                let (chunk_verts, chunk_idxs) = r.chunk_buffer_stats();
                let streamer_stats = self
                    .world_state
                    .streamer
                    .as_ref()
                    .and_then(|s| s.stats())
                    .unwrap_or_default();
                let ecs = self.simulation.ecs_world();
                let snap = crate::telemetry::TelemetrySnapshot {
                    time: self.telemetry.elapsed_secs(),
                    cpu_frame_ms: (self.input.frame_time * 1000.0) as f32,
                    gpu_frame_ms: gpu.frame_ms,
                    gpu_shadow_ms: gpu.shadow_ms,
                    gpu_sky_ms: gpu.sky_ms,
                    gpu_opaque_ms: gpu.opaque_ms,
                    gpu_transparent_ms: gpu.transparent_ms,
                    gpu_ui_ms: gpu.ui_ms,
                    gpu_post_ms: gpu.post_ms,
                    gpu_allocated_mb: alloc_bytes as f32 / 1_048_576.0,
                    gpu_reserved_mb: reserved_bytes as f32 / 1_048_576.0,
                    process_rss_mb: {
                        // Refresh sysinfo every 60 frames to avoid overhead.
                        self.sysinfo_refresh_counter += 1;
                        if self.sysinfo_refresh_counter >= 60 {
                            self.sysinfo_refresh_counter = 0;
                            self.sysinfo
                                .refresh_processes(sysinfo::ProcessesToUpdate::All, false);
                        }
                        let pid = sysinfo::get_current_pid().unwrap_or(sysinfo::Pid::from_u32(0));
                        self.sysinfo
                            .process(pid)
                            .map(|p| p.memory() as f32 / 1_048_576.0)
                            .unwrap_or(0.0)
                    },
                    chunks_loaded: self.world_state.world.loaded_chunk_count() as u32,
                    chunks_meshed: self.world_state.world.meshed_chunk_count() as u32,
                    chunks_gpu: r.chunk_count() as u32,
                    chunk_vertices: chunk_verts,
                    chunk_indices: chunk_idxs,
                    streamer_gen_queue: streamer_stats.gen_queue,
                    streamer_mesh_queue: streamer_stats.mesh_queue,
                    streamer_pending_remesh: streamer_stats.pending_remesh,
                    streamer_gen_ms: streamer_stats.gen_ms,
                    streamer_mesh_ms: streamer_stats.mesh_ms,
                    water_tick_ms,
                    water_pending_flow: self.world_state.world.pending_flow_count() as u32,
                    chunk_upload_ms,
                    entity_count: ecs.entity_count(),
                    archetype_count: ecs.archetype_count() as u32,
                };
                self.telemetry.push(snap);
            }
        }

        // Screenshot requested via keybind (deferred from event handler).
        if self.input.screenshot_requested {
            self.input.screenshot_requested = false;
            self.do_capture();
        }

        // Auto-capture for verification.
        self.input.frame_count += 1;
        if let Some(after) = self.config.capture_after_frames {
            if !self.input.captured && self.input.frame_count >= after {
                self.input.captured = true;
                self.do_capture();
            }
        }
    }

    /// Render a still frame to PNG via the auto-capture machinery.
    pub(crate) fn do_capture(&mut self) {
        let mut camera = self.simulation.player_camera().unwrap_or_default();
        let h = self.render.window_size.1 as f32;
        camera.aspect = if h > 0.0 {
            self.render.window_size.0 as f32 / h
        } else {
            1.0
        };
        let ui = self.build_ui();
        let dp = self.day_params();
        let eye_block = voxel_core::math::world_to_block(camera.pos);
        let underwater = self
            .world_state
            .world
            .is_liquid(eye_block.x, eye_block.y, eye_block.z);
        let Some(r) = self.render.renderer.as_mut() else {
            return;
        };
        r.set_sky(
            dp.horizon,
            dp.zenith,
            dp.fog,
            dp.daylight.max(0.15),
            underwater,
        );
        r.set_reflection_strength(self.config.reflection_strength);
        {
            let ext = r.extent();
            r.set_proj_params(camera.near, camera.far, ext.width as f32, ext.height as f32);
        }
        match r.capture_frame(voxel_render::FrameInput {
            camera,
            ui: Some(&ui),
            game_time: self.gameplay.game_time as f32,
            underwater,
            world_entities: &[],
            held_items: &[],
            show_panorama: false,
            panorama_rotation: 0.0,
        }) {
            Ok(rgba) => {
                let (w, h) = self.render.window_size;
                if let Some(img) = image::RgbaImage::from_raw(w, h, rgba) {
                    match img.save(&self.config.capture_path) {
                        Ok(()) => log::info!(
                            "captured frame to {} ({}x{}, chunks={})",
                            self.config.capture_path.display(),
                            w,
                            h,
                            r.chunk_count()
                        ),
                        Err(e) => log::error!("save capture: {e}"),
                    }
                } else {
                    log::error!("capture: image size mismatch, failed to create RgbaImage");
                }
            }
            Err(e) => log::error!("capture_frame: {e}"),
        }
    }
}

/// Compute cascaded shadow-map view-projection matrices for the four cascades
/// used by the renderer. Each cascade projects the camera frustum sub-frustum
/// from the sun's POV.
impl crate::EngineApp {
    /// Build the [`voxel_render::MaterialTable`] consumed by the chunk
    /// fragment shader's binding 5 (leaves SSS + wet-edge tint + sun caustics).
    ///
    /// Each `BlockDef` contributes its `material` field to every face-tile
    /// index it owns (six entries from `BlockTextures::tiles`). When two
    /// blocks reference the same texture tile, the LATER block definition
    /// wins; in practice this only happens for special-cased "shared-tile"
    /// blocks like `light_pink` and friends, and the resulting visual is
    /// indistinguishable from the per-block ownership we already encode.
    ///
    /// `water_y / wet_edge / caustics / leaves` are the global master
    /// scalars from `GraphicsSettings`, packed into `world_params.wxyz`
    /// matching the GLSL std430 layout in `shaders/chunk.frag`.
    pub(crate) fn build_material_table_from_registry(
        reg: &voxel_world::BlockRegistry,
        water_y: f32,
        wet_edge: f32,
        caustics: f32,
        leaves: f32,
    ) -> voxel_render::MaterialTable {
        use voxel_render::{BlockMaterialGpu, MaterialTable, TILE_MATERIAL_TABLE_LEN};
        let mut table = MaterialTable::empty();
        let count = reg.count();
        for i in 0..count {
            let def = reg.get(voxel_core::BlockId(i as u16));
            let m = def.material;
            let gpu = BlockMaterialGpu::pack(
                m.flags,
                m.roughness,
                m.emissive,
                m.sss_tint,
                m.wet_tint,
                m.absorption,
            );
            for &tile in &def.textures.tiles {
                let idx = tile as usize;
                if idx < TILE_MATERIAL_TABLE_LEN {
                    table.materials[idx] = gpu;
                }
            }
        }
        table.world_params = [water_y, wet_edge, caustics, leaves];
        table
    }
}

fn compute_shadow_cascades(
    camera: &voxel_core::Camera,
    sun_dir: glam::Vec3,
    near: f32,
    far: f32,
) -> ([[f32; 16]; 4], [f32; 4], [f32; 4]) {
    let split_factors = [0.05, 0.15, 0.4, 1.0];
    let mut cascade_vps = [[0.0f32; 16]; 4];
    let mut cascade_splits = [0.0f32; 4];

    let view_proj = camera.view_projection();
    let inv_vp = view_proj.inverse();

    let sun_dir = sun_dir.normalize();

    for i in 0..4 {
        let prev_split = if i == 0 { 0.0 } else { split_factors[i - 1] };
        let split = split_factors[i];

        let near_split = near + (far - near) * prev_split;
        let far_split = near + (far - near) * split;
        cascade_splits[i] = far_split;

        let corners_ndc = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];

        let mut corners_world = [glam::Vec3::ZERO; 8];
        for j in 0..8 {
            let ndc_z = if j < 4 {
                -1.0 + 2.0 * (near_split - near) / (far - near)
            } else {
                -1.0 + 2.0 * (far_split - near) / (far - near)
            };
            let ndc = glam::Vec4::new(corners_ndc[j][0], corners_ndc[j][1], ndc_z, 1.0);
            let world = inv_vp * ndc;
            corners_world[j] = world.truncate() / world.w;
        }

        let center = corners_world
            .iter()
            .fold(glam::Vec3::ZERO, |acc, &c| acc + c)
            / 8.0;

        let mut min = corners_world[0];
        let mut max = corners_world[0];
        for &c in &corners_world[1..] {
            min = min.min(c);
            max = max.max(c);
        }

        let light_pos = center - sun_dir * 100.0;
        let up = if sun_dir.y.abs() > 0.99 {
            glam::Vec3::new(1.0, 0.0, 0.0)
        } else {
            glam::Vec3::new(0.0, 1.0, 0.0)
        };
        let light_view = glam::Mat4::look_at_lh(light_pos, center, up);

        let mut light_min = glam::Vec3::ZERO;
        let mut light_max = glam::Vec3::ZERO;
        for &c in &corners_world {
            let lc = light_view * glam::Vec4::new(c.x, c.y, c.z, 1.0);
            let lc = lc.truncate();
            if lc.x < light_min.x {
                light_min.x = lc.x;
            }
            if lc.x > light_max.x {
                light_max.x = lc.x;
            }
            if lc.y < light_min.y {
                light_min.y = lc.y;
            }
            if lc.y > light_max.y {
                light_max.y = lc.y;
            }
            if lc.z < light_min.z {
                light_min.z = lc.z;
            }
            if lc.z > light_max.z {
                light_max.z = lc.z;
            }
        }

        let pad = 2.0;
        light_min -= glam::Vec3::splat(pad);
        light_max += glam::Vec3::splat(pad);

        let radius = (light_max - light_min).length() * 0.5;

        let light_proj = glam::Mat4::orthographic_lh(
            light_min.x,
            light_max.x,
            light_min.y,
            light_max.y,
            -radius - 50.0,
            radius + 50.0,
        );

        let vp = light_proj * light_view;
        cascade_vps[i] = vp.to_cols_array();
    }

    let light_dir_and_bias = [sun_dir.x, sun_dir.y, sun_dir.z, 0.01];

    (cascade_vps, cascade_splits, light_dir_and_bias)
}
