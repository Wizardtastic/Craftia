# Game GUI Toolkit

How the game GUI is built: a procedural chrome atlas, a thin immediate-mode
widget layer (`ui_kit`), and per-screen `draw_*` functions in
`crates/engine/src/ui.rs`. No GUI dependencies — everything is generated at
startup and drawn as quads through the existing UI pipeline.

## Architecture

```
procedural atlas (voxel-render/src/ui_atlas.rs)
  └─ uploaded once at startup as the 4th UI sampler (tex_id = 3)
      └─ UiDrawData helpers (voxel-render/src/ui.rs)
            sprite / sprite_wh / nine_slice / gradient_v / gradient_h
            text_shadow / progress_bar
          └─ UiKit widgets (voxel-engine/src/ui_kit.rs)
                button, small_button, panel, panel_inset, slot, sel_frame,
                slider, toggle, scrollbar, label, tooltip, dim
              └─ per-screen draw_* functions (voxel-engine/src/ui.rs)
```

- **`ui_atlas.rs`** generates every chrome tile on the CPU at startup:
  beveled buttons (normal/hover/pressed/disabled), inset slots, panels,
  tooltips, scrollbar pieces, the crosshair, and pixel-art HUD icons
  (hearts, hunger, armor, air bubbles). Chrome tiles are grayscale so the
  per-vertex tint picks the widget colour at draw time.
- **`UiDrawData` helpers** are the renderer-facing primitives. `nine_slice`
  stretches a tile as a 9-patch with a 4 px border; corners keep their
  aspect, edges stretch, the centre fills. `text_shadow` draws the classic
  dark-offset game text.
- **`ui_kit::UiKit`** is a stateless immediate-mode layer. Widgets return
  hit-test-ready `Rect`s, so click handlers either use the returned rect
  directly or consume a shared layout (see below). The remaining stored-rect
  fields are gone; new screens must not add any.
- **`ui_kit::Scroll`** is the lightweight scrolling helper for lists longer
  than the visible rows: `Scroll::new(row, total, visible)` clamps the
  offset, `scroll_by` applies wheel deltas, `scrolled_to_show` keeps a
  selection visible (keyboard nav), and `thumb()`/`visible_fraction()` feed
  `UiKit::scrollbar`. World select uses it; reuse it for any new list.
- **`screen_layout.rs`** holds pure per-screen layout structs
  (`TitleLayout`, `PauseLayout`, `SettingsLayout`, `WorldSelectLayout`,
  `CreateWorldLayout`, `DeleteConfirmLayout`) built from the window size
  (+ small flags). The draw pass renders *from* the layout; the click /
  drag / key handlers rebuild it and hit-test *against* the same rects.
  No widget rects are stored between frames.
- **Contrast policy** (the Minecraft look): chrome tiles are bright gray
  (button fill ≈ 0.8, panel ≈ 0.78), so text on buttons and panels uses the
  dark `palette::INK` family (`INK`, `INK_MUTED`, `INK_DISABLED`,
  `INK_DANGER`). The `TEXT`/`MUTED` whites are only for genuinely dark
  surfaces: inset panels, text fields, chat, console, tooltips.
- **Screen functions** compose the above. Draw order = z order; the tooltip
  queue is flushed last via `kit.finish()`.

## Adding a new screen

```ignore
fn draw_my_screen(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
    let mut kit = crate::ui_kit::UiKit::new(
        ui,
        &self.render.font,
        (self.gameplay.mouse_pos.x, self.gameplay.mouse_pos.y),
    );
    kit.dim(0.0, 0.0, w, h, 160);                 // veil over the world
    kit.panel(40.0, 40.0, 300.0, 200.0,
              [205, 205, 215, 255]);              // dialog background
    kit.label("Title", 56.0, 52.0, 1.2,
              [255, 255, 255, 255]);
    let r = kit.button(70.0, 120.0, 240.0, "DO IT");
    kit.tooltip("Does the thing", r.x, r.y + r.h + 4.0);
    kit.finish();                                  // flush tooltips last
}

// Click handler: rebuild the SAME layout and hit-test it — no stored state.
fn handle_my_screen_click(&mut self) {
    let lay = crate::screen_layout::MyScreenLayout::new(w, h);
    if lay.do_it_btn.contains(self.gameplay.mouse_pos) {
        /* ... */
    }
}
```

Rules of thumb:

1. **Never duplicate layout math** between draw and click handlers — put it
   in a `screen_layout` struct (or return the rect from one kit call) and
   share it.
2. Keep chrome grayscale in the atlas; tint per widget at draw time.
3. All layout happens in logical pixels; the pipeline scales by
   `ui_scale` before upload.
4. Text uses `text_shadow` (or `kit.label`) unless it sits on a light
   background.

## Testing

`crates/engine/src/ui_kit.rs` has layout unit tests: hit-box/draw parity,
9-slice quad coverage (including degenerate small rects), slider/toggle
geometry, scrollbar clamping, `Scroll` windowing/selection/empty-list
invariants, and font/atlas generation invariants. Run with
`cargo test -p voxel-engine ui_kit`.

`crates/engine/src/screen_layout.rs` adds pure-geometry tests for the
title, pause, settings and world-select screens: row-list/rect parity,
panel containment, slider value mapping, visible-row capping, scroll
windowing, and dialog button overlap.

Golden-image verification: `cargo test -p voxel-app --test snapshots`
(delete `tests/snapshots/golden_1337.png` to regenerate after intentional
visual changes).
