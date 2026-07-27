# Spec: Water & Translucency Upgrade

> Short name: **water-translucency**. Status: **draft, awaiting sign-off**.
> Goal: take the engine from "looks flat / Minecraft-y" toward a stylized PBR-leaning
> voxel renderer that still feels like a voxel game — while staying at ~60 fps on a
> GTX 1060 + Ryzen 3600X and remaining a healthy long-term foundation.

---

## 1. Background (where we are today)

A quick read of what we already have, so the spec is grounded in real code paths.

* **Renderer:** Forward. Two visible meshes per chunk (`opaque`, `transparent`), one
  combined draw inside a single Vulkan render pass (see `crates/render/src/renderer.rs`,
  `record_chunk_passes`). Position+UV+AO+torchlight+tile are baked into the vertex
  stream in `crates/world/src/mesh.rs`.
* **Lighting model:** Baked per-vertex "frag_light ∈ [0, 1]" plus a saturated coloured
  torchlight tint (`shaders/chunk.frag`). The fragment shader does **not** have any
  surface normal — every voxel face shades identically until you add `frag_light_color`
  on top. No specular. No roughness.
* **Sky:** simple panorama shader.
* **Shadowing:** 4-cascade PCF sun shadow, 2048² default, normalised in
  `shaders/chunk.frag::compute_shadow_factor`. No shadow caster baking for torches.
* **Post chain (`shaders/post.frag`):**
  * SSAO (32-tap hemisphere kernel, reconstructed normal from depth gradients)
  * naïve threshold bloom (3 rings, 4 taps each, very visible ringed halo)
  * ACES tonemap
  * vignette
  * underwater distortion (cheap screen-space wobble + blue/green tint, no fog
    volume, no caustics)
* **Scene render target:** appears to be sampled and bloomed before tonemap, but
  the bloom and exposure code path implies scene_color holds values that are
  effectively LDR-ish (no FP16 framebuffer in evidence).
* **Water:** flat semi-transparent quad. `crates/world/src/mesh.rs::emit_water_block`
  builds a quad whose top y matches the water level. No reflections, no refraction,
  no caustics, no foam. Underwater look is performed entirely in `post.frag` as a
  screen-wide tint.
* **Translucency today:** leaves, tall grass, glass-like blocks use
  `if (tex.a < 0.1) discard;` (alpha cutout). That handles silhouettes but does not
  give coloured backlight, refraction, or stained-glass character.
* **Save format:** chunk saves already bumped to **v3**, which stores the packed
  RGBA8 `torchlight_color` array per voxel (`crates/world/src/save.rs`). We have
  a clean migration hook for **v4**.
* **Texture sampling:** atlas sampler is **NEAREST + CLAMP** — the comment in
  `shaders/chunk.frag` explicitly forbids linear filtering for seam reasons. We
  will not break this contract.
* **Performance budget:** current fps headroom on a 1060/3600X at 1280×720/MSAA-4
  is comfortable; we have ~6-8 ms GPU to play with before we clip 60 fps.

---

## 2. Goals & non-goals

### Goals

1. **Water stops being the worst thing on screen.**
   * Sky/sun reflection (probe).
   * SSR for nearby world reflection (mountains, walls, towers).
   * Refraction with Beer's-law-style colour attenuation with depth.
   * Procedural sun caustics on submerged terrain.
   * Shore foam / wet edges.
2. **Translucency actually looks translucent.**
   * Leaves: coloured backlight (S/CSS-flavored).
   * Glass: real transparency + refraction sampling the previously-rendered
     opaque scene.
   * Stained glass: per-block colour emits a tinted light onto nearby surfaces.
3. **Lava gets its own emissive, animated, light-casting treatment.**
4. **Underwater view becomes a volume, not a post-process hack.**
   * Caustics on submerged surfaces.
   * Vertex-distorted water plane stays.
   * Volumetric colour fog growing with depth (real depth fog, not the blue tint).
   * Optional god rays (procedural screen-space shafts).
5. **The architecture is forward-friendly and survives future additions** —
   any future water/glass/sky/translucent feature should compose with this
   instead of going around it.

### Non-goals (this round)

* Full-blown PBR (no metallic-roughness textures per voxel atlas tile; we use a
  scalar roughness + block-tinted emission as a proxy).
* Animated blocky leaves (mesh is unchanged; effects are shader-side).
* Skybox HDR procedural scattering from scratch. We reuse the panorama + a new
  procedurally-synthesized cube probe.
* Vulkan ray-traced reflections.
* Touching the deferred / vertex-skinning / model-loaded entity paths beyond
  what they already do.
* Rewriting the asset pipeline; new normal-map tiles slot into the existing
  atlas.

---

## 3. Architecture decision: **restructure forward (Option B)**

We considered deferred rendering. We chose **not** to go deferred. Reasons:

* **Voxel games aren't light-bound, they're polygon-bound.** The engine already
  encodes per-voxel torchlight + colour into the mesh. The "thousands of
  dynamic point lights" problem that deferred solves simply doesn't apply.
* **Transparency is the dominant case.** Water, glass, stained glass, leaves,
  lava, particles, UI, the editor overlay — *all* the things we are upgrading
  benefit from forward blending. In a deferred path, every one of these still
  needs a forward pass, so the G-buffer we'd write is mostly cost without payoff.
* **Bandwidth on a 1060.** A 4-channel G-buffer doubled with depth is heavier
  per frame than two reorganized forward passes that reuse the depth we already
  have.
* **Forward is the natural place to compute screen-space effects.** SSR,
  refraction, depth-fog, and god rays all read the (already written) opaque
  depth + scene colour. Keeping the pipeline forward lets us share those
  buffers naturally.

### What changes in the pass layout

**Today (one render pass, everything inside):**
```
OPAQUE chunks  →  TRANSPARENT chunks  →  particles →  UI (into scene_color)
                                              ↓
                                       post.frag (SSAO, bloom, tonemap)
```

**After (split for shared-buffered transparent work):**
```
1. OPAQUE pass       — world, entities, sky(under clear), to offscreen_image + depth
2. Copy/Barrier      — blit scene_color → scene_opaque_color (sampled)
                        transition depth → shader-readable (sampled)
3. TRANSPARENT pass  — water, glass, leaves, lava, stained glass, to scene_color
                        (samples scene_opaque_color + depth for refraction, foam)
4. PARTICLES pass    — as today
5. UI pass           — as today
6. POST pass         — SSAO, god rays, depth fog, bloom, tonemap
```

The big change is **separating opaque and transparent into different render
passes** so transparent code can sample the opaque scene colour and depth. We
do **not** introduce a deferred-lighting G-buffer; this is still forward.

---

## 4. Water design

### 4.1 Surface mesh & normals

* Keep the existing top-quad per water level from `emit_water_block`.
* Add a **procedural wave normal** in the fragment shader:
  `n = normalize(grad_height + dFdx + dFdy)`, with two octaves of sine-driven
  displacement. Subsurface "wave crests" come from a sin/cos pair driven by
  `time + world.xz * frequency`.
* On top of that, **blend a baked normal-map tile** sourced from the existing
  atlas (one new tile). The blend ratio is configurable; default ~0.4. This
  gives stylistic "high-frequency facet noise" that survives without animation.
* The wave normal is also what drives reflection refraction direction in SSR
  and screen-space refraction, so the surface actually *looks* curved.

### 4.2 Reflections

Two contributions, composited:

* **Sky/sun probe** — a small (e.g. 64²) cubemap baked **per frame** from the
  existing panorama shader plus the current sun direction. We already render
  `sky.frag`; we add a one-second-per-frame probe render target that re-uses
  the same shader with a face index. Cost: ~6 draws per frame at 64² — trivial
  on a 1060. This is the dominant reflection (99 % of pixels).
* **Screen-space reflection (SSR)** — a small per-pixel ray-march against
  `scene_opaque_color` + `depth` with a hard cap of e.g. 24 steps and a
  thickness test (linearised depth difference ≤ ε). Strength falls off with
  distance and roughness. Returns the sky probe colour on ray miss.

SSR is the most expensive piece; the budget is ~0.6 ms (educated guess; exact
TBD during implementation, see §8).

### 4.3 Refraction & depth attenuation

* Refraction UV offset = surface normal.xy × `refraction_strength` ×
  `1.0 / view_depth` (curvature-correct for perspective).
* Beer's-law attenuation: `tint = exp(-absorption_coeff × depth_through_water)`
  where `depth_through_water` = linearised depth minus camera-water surface y
  (clamped ≥ 0), and `absorption_coeff` is per-block (registry-driven, default
  `(0.45, 0.18, 0.10)` — water-blue).
* `absorption_coeff` lives in the registry (no save bump required for water).
* Result mixes with the (refracted) opaque scene colour.

### 4.4 Sun caustics under water

* Sun direction already lives in `fog.ambient_and_sun.yzw`. We use it directly.
* Caustic pattern: a quad of `sin/cos` cross-products warped by caustic
  Voronoi-ish fudge on (world.xz, time). It's procedural — no texture needed.
* Visible on terrain below water: applied in the **chunk shader** as a
  multiplicative additive term when a block is below the local water level AND
  receives direct sunlight. This lives in chunk.frag so we don't need a second
  pass.

### 4.5 Shore foam & wet edges

* Where a water block is adjacent on the XZ axes to a non-water solid block,
  emit foam quads into the **transparent** mesh (new sample helper in
  `crates/world/src/mesh.rs`).
* Foam normal: same procedural wave normals; foam opacity falls with distance
  from the contact edge (≤ 0.5 voxel ideal, configurable).
* Foam colour: warm white with a gentle blue fresnel rim. As a quick win,
  foam tint is configurable in `config.toml` so we don't ship a hardcoded blue.
* "Wet edge": in the chunk shader, when a non-water opaque face is horizontally
  adjacent to a water source block (level ≥ 7), add a soft `+0.15` * a small
  `vec3(0.85, 0.92, 1.0)` tint (cool, dampened) to its final colour, with the
  effect fading over `1.0` voxel from the contact line. Cheap and very effective.

---

## 5. Translucency design

### 5.1 Leaves (subsurface backlight)

* Leaves already discard on `tex.a < 0.1`. We extend without changing the mesh.
* In `chunk.frag`, when the block's tile index falls into the configured
  `leaves_tile_range` (registry-driven, default `[leaves]`), compute:
  `back_light = max(dot(-sun_dir, face_normal), 0.0)` using `dFdx/dFdy`-derived
  flat normal (we don't currently have a real normal — derive it cheaply from
  position derivatives).
  `final_rgb = mix(leaf_rgb, leaf_rgb * (1.5, 2.2, 1.2), 0.5 * back_light)`
  (slightly green-yellow tinted, mimicking chlorophyll under backlight).
* Leaves also benefit from a small ambient occlusion bias (already present).

### 5.2 Glass & stained glass

* Glass tiles stop using `discard`. They use a real transparent pipeline path
  that backs onto the **scene_opaque_color** sample for refraction.
* Block registry gets a new field `translucency_tint: [u8; 3]` (default
  `(255, 255, 255)` for clear glass; warm/cool tints for tinted variants).
* Per-block: `refraction_strength: f32` (default 0.015 for glass, 0.0 for
  leaves), `roughness: f32` (frag-fog-overridden).
* Stained glass **emits coloured light** onto adjacent surfaces: at registration
  time, a `stained_glass: bool` flag. When true, in the *receiving* block's
  shading we add `tint × neighbor_tint × neighbour_emission_strength`. This is
  computed in the **per-block vertex** stage now that we have a way to encode
  light colour — see §7.

### 5.3 Practical notes

* Glass blocks with high refraction need MSAA ≥ 4 — we already have it.
* List of glass-like tile ranges is registry-driven; default seeds compile in
  `crates/world/src/registry.rs` (clear glass, stained glass × 3 colours).

---

## 6. Lava

* Lava is a **separate transparent material** that lives in the chunk shader's
  transparent branch alongside water.
* Same wave-normal machinery as water, but lower amplitude, warmer colour,
  and a faster temporal drift (animated "blob" pattern).
* Lava is treated as a **strong emitter**: surface emission `~3.0` (clamped to
  HDR range) fed into bloom + scene_opaque_color. Drag the `Lava` slider in the
  settings UI to control intensity.
* Lava emits warm light onto neighbouring blocks exactly like torchlight,
  but its colour comes from the **lava tile** itself (`registry::lava_color`).
* No vertical-flow reflection — lava is opaque to anything beneath it in this
  pass (cheaper, looks good).

---

## 7. Data model changes (chunk save v4)

We are bumping the chunk save format to **v4** with a graceful auto-translator.

### New per-block fields (per voxel, mirrors the existing v3 torchlight_color layout)

| Field                     | Type    | Default         | Meaning                                              |
| ------------------------- | ------- | --------------- | ---------------------------------------------------- |
| `material_flags`          | u8      | 0               | bit0 = translucent_sss, bit1 = stained_glass, …       |
| `roughness`               | u8      | 128 (≈ 0.5)     | 0 = mirror, 255 = matte                              |
| `emissive`                | u8      | 0               | 0 = none, 255 = full emissive                        |
| `translucency_tint`       | u32     | 0xFFFFFFFF      | packed RGBA8                                         |

**Defaults are registry-driven** so the engine doesn't ship a half-broken v4
save. When we read a v3 chunk, we fill in default values from the block ID →
registry lookup. When we read a v2 or earlier chunk, we go through the same
fill path (already in v3).

### Schema

```rust
// crates/world/src/chunk.rs (new fields on Chunk, parallel to torchlight_color)
pub(crate) material_flags: Box<[u8]>,        // CHUNK_CUBED
pub(crate) roughness: Box<[u8]>,            // CHUNK_CUBED
pub(crate) emissive: Box<[u8]>,             // CHUNK_CUBED
pub(crate) translucency_tint: Box<[u32]>,   // CHUNK_CUBED
```

### Saves

* v4 layout writes the new arrays after `torchlight_color`. v3 reads transparently
  (defaults applied per registry). v4 writes new voxels with registry-driven
  defaults.
* `crates/world/src/save.rs::chunk_v4_header` gains a `format_version = 4` and
  the new `material_data()` slice.

### Editor (`edit::paint`)

While painting a block, surface the new fields if the tile belongs to a family
that supports them (e.g. stained glass palette). Default copy from registry.

---

## 8. Render pipeline: pass-by-pass budget

Approximate per-frame cost at 1280×720 (4× MSAA) on a GTX 1060; budgeting for
60 fps with a 16.6 ms envelope. Current baseline (read-only) is ~5 ms
total GPU, so we have ~10 ms of headroom.

| Pass                        | Cost (target)  | Notes                                         |
| --------------------------- | -------------- | --------------------------------------------- |
| Opaque (chunks, entities)   | +0.1 ms        | unchanged                                     |
| **Cube probe render (sky)** | ~+0.2 ms       | 6×64² draws/frame                             |
| **Copy/Barrier**            | ~0.05 ms       | single blit + layout transition               |
| **Transparent (water/glass)**| +0.6 ms        | wave normals, refraction sample, SSR march    |
| **SSR march (within transparent)** | ~0.4 ms   | clamped 24 steps, early-out                   |
| Particles                   | unchanged      |                                               |
| UI                          | unchanged      |                                               |
| Post (extended)             | +0.2 ms        | god rays (cheap screen-space), depth fog ok   |
| **Total new**               | **~+1.55 ms**  |                                               |

Hard guardrails:

* SSR ray step count capped at 24.
* Probe res capped at 64² (configurable: 32, 64, 128).
* HDR optional via `config.toml`; `scene_color` is FP16 RGBA when enabled.
* HDR-off path still produces bloom & specular, but everything clamps to ≤1.0
  *before* the bloom downsampled reads. This is the safe fallback for GPUs
  that don't have FP16 blending cheaply (rare on 1060).
* Quality profiles (Low/Med/High) gate: cube probe res, SSR steps, refraction,
  god rays.

---

## 9. Configuration & UI

New entries in `config.toml`:

```toml
[graphics]
hdr_scene = true                 # FP16 scene_color if true
water_probe_resolution = 64      # 32 | 64 | 128
water_ssr_steps = 24             # 8..32
water_ssr_strength = 0.6         # 0.0..1.0
water_caustics_strength = 0.5    # 0.0..1.0
foam_enabled = true
wet_edge_strength = 0.15
stained_glass_emission = 0.6
lava_emission = 2.0
god_rays_strength = 0.0          # off by default; toggleable at runtime
quality_profile = "high"         # "low" | "med" | "high"
```

UI in `crates/engine/src/ui.rs`:

* New left-panel graphics additions (mirrors the existing SSAO panel).
* Quick toggles:
  * "Reflections" (sets probe + SSR strength scalar)
  * "Refraction"
  * "Caustics"
  * "Foam / Wet edges"
  * "Stained glass emission"
  * "Lava emission"
* `F3` debug overlay gets a new line: `Water: probe=64², ssr=on, refraction=on`.

Editor mode (`X` key) inherits the same toggles — same long-term engine
foundation principle. Editor preview should still show the visual upgrades.

---

## 10. Files we'll touch (rough inventory)

* `crates/render/src/renderer.rs`
  * Split opaque/transparent pipeline bindings; add `cubemap_probe_pass`.
  * Add `scene_opaque_color` image + `depth_sampled` view.
* `crates/render/src/renderer/pipeline.rs`
  * New pipeline records for water/glass (`PipelineKind::TranslucentV2`),
    cube-probe (`PipelineKind::SkyCubeProbe`).
* `crates/render/src/texture.rs` — no functional change; comments only
  (NEAREST/CLAMP contract is preserved for atlas sampling).
* `crates/world/src/chunk.rs`
  * New `material_flags`, `roughness`, `emissive`, `translucency_tint`
    arrays; helper setters; default-on-init paths.
* `crates/world/src/save.rs`
  * Bump to v4 (`chunk_v4_header`); write/read new fields; legacy translators.
* `crates/world/src/mesh.rs`
  * Add `emit_foam_quads_for_water_chunks`.
  * Refactor `transparent` mesh into **Lava** and **Water+Glass** subbundles
    so each can have its own pipeline binding without losing the existing
    sort-by-camera-distance pass.
* `crates/world/src/registry.rs`
  * New block-level fields: `roughness`, `emissive`, `translucency_tint`,
    `refraction_strength`, `is_leaves`, `is_stained_glass`,
    `absorption_coeff`. Defaults for glass/clear/stained.
* `crates/world/src/light.rs`
  * Stained-glass emission propagation: small extra BFS contribution that
    scales neighbour `torchlight_color` by glass tint.
* `shaders/chunk.frag`
  * New material branch — leaves SSS, glass refraction sample,
    stained glass receive, wet-edge tint, sun caustics additive when below
    water, lava receive.
* `shaders/water.frag` *(new)*
  * Wave normals, refraction sample, SSR ray-march, foam composite,
    sun glint, sky probe fallback.
* `shaders/glass.frag` *(new — or merged into chunk.frag if cheap)*
  * Transparency blend, refraction sample, stained-glass tint.
* `shaders/sky.frag`
  * New `PROBE_FACE_INDEX` push constant for cube-probe rendering; emissive
    "fog/sky-cooler" near horizon for water to reflect.
* `shaders/post.frag`
  * Add god rays (cheap screen-space shafts sampled off a low-res depth
    sun-mask pre-pass OR a depth-derivative estimate). Add volumetric
    underwater fog mixing SSR-style. The existing underwater tint block
    goes away.
* `crates/engine/src/ui.rs` — new panel + per-feature toggles.
* `crates/engine/src/frame.rs` — new feature toggles reach the renderer.
* `crates/engine/src/settings.rs` — new fields with sane defaults.
* `crates/engine/src/lib.rs` (`ConfigRender`) — new fields and defaults.
* `config.toml` — shipped defaults.
* `crates/render/tests/smoke.rs` — new pass coverage tests.
* `docs/notes/` — `u2014_water_translucency.md`: a short technical note
  describing the pass split and the SSR ray-march approach.

---

## 11. Acceptance criteria

A reviewer should be able to look at the screen and say "yes" to all of these:

* Water surface looks "wet": visible sky-colour reflection, sun glint, moving
  wave normals, refraction of submerged geometry.
* Waves are visible parallax (object placed mid-water shifts with the surface
  normal on the screen).
* Caustics: sun-lit submerged terrain near water surface has moving bright
  spots.
* Foam lines visible where water meets solid blocks; gradient tapers over
  ~0.5 voxel.
* Wet edge: a 1-voxel wet rim of cool tint on solid blocks touching water.
* Glass blocks are properly transparent (you can see through them with
  refraction); stained glass tints light onto nearby surfaces.
* Leaves show coloured backlight when sun is behind them.
* Lava is visibly hot, animated, casts warm light onto neighbouring surfaces,
  and its emission contributes to bloom.
* Underwater view: depth-tinted volumetric fog, real caustics on terrain,
  vertex-distorted water surface, optional god rays from the sun.
* Performance: live metric from `F3` (`GpuTransparentMs`) ≤ ~1.5 ms at
  1280×720 on a 1060, total frame ≤ 12 ms.
* Save format: loading an existing v3 save gives a visually identical world
  (with default material values populated). Loading a v2 save still works;
  loading v1 works via existing v2 translation.

---

## 12. Risks & open questions

* **Risk:** SSR edge leaking on hi-contrast cliff edges. *Mitigation:* thickness
  test, early-out, strength fall-off with distance. *Test before committing.*
* **Risk:** FP16 blending cost on hardware that lacks it cheaply (older
  integrated GPUs). *Mitigation:* `hdr_scene = false` falls back gracefully.
* **Risk:** Wave-normal computation coupled with refraction may cause visible
  swimming artifacts. *Mitigation:* ray-march step length bounded; SSR step
  count capped.
* **Risk:** Material defaults could regress visual feel on existing saves.
  *Mitigation:* all defaults pulled from the registry on v3→v4 migration; v3
  loads are byte-identical to today's visual save.
* **Risk:** Adding cubemap probe + scene-opaque copy increases references and
  may make descriptor-set reuse harder. *Mitigation:* migrate to a shared
  "scene" descriptor layout across all post passes.
* **Open:** Should stained-glass propagate up to the BFS torchlight (cheap
  but has reach limits), or only reach via vertex AO bake? Pick one during
  implementation; default to BFS propagation for consistency with torches.

---

## 13. What's deliberately left for a follow-up spec

* Day/night cycle polish (atmospheric MIE scattering).
* Real subsurface scattering for flesh/wool.
* Particle-system rewrite for smoke/fire (lava smoke effect).
* Volumetric cloud rendering.
* TAA / SMAA pass to fight aliasing on distant geometry. (Atlas NEAREST is
  required for the seam fix; we won't change that. TAA motion vectors would
  be a clean fix for shimmer.)

---

## 14. Quick map of decisions back to the interview

| Question                                                | Decision                                                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| Scope?                                                  | Medium subsystem upgrade                                                |
| Driver?                                                 | Looks too flat / Minecraft-y → fight flatness                            |
| Audience?                                               | Long-term engine foundation                                              |
| Subsystem?                                              | Water & translucency (with full feature stack)                          |
| Style?                                                  | Stylized PBR (Minecraft RTX-ish)                                         |
| Performance?                                            | 60 fps on 1060/3600X with HDR optional + quality profile                   |
| Water features?                                         | Probe + SSR + refraction (depth attenuation) + caustics + foam/wet edge  |
| Translucency?                                           | Leaves SSS + glass refraction + stained glass                             |
| Underwater?                                             | God rays + sharper distortion + colour fog depth + bubble particles       |
| Lava?                                                   | Include                                                                  |
| Architecture?                                           | Restructure forward pipeline (Option B) — **NOT** deferred                |
| Data model?                                             | Save v4 (chunk save bump) + registry defaults (hybrid)                    |
| Wave normals?                                           | Procedural wave normals + baked normal-scale overlay                      |
| HDR target?                                             | Optional (config + UI), default on                                      |
