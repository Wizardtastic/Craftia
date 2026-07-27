# `\`u2014` (legacy 4-digit) vs `\`u{2014}` (curly) vs raw UTF-8 bytes — empirical matrix

**TL;DR.** rustc 1.96 rejected the legacy 4-digit unicode escape `` `\u2014` `` inside two test panic strings in `crates/engine/src/lib.rs` with `error: incorrect unicode escape sequence`. The curly form `` `\u{2014}` `` compiles cleanly in the same context, so that's the form we ship. Raw UTF-8 em-dash bytes (`E2 80 94`) in the source are also fine and serve as a fallback.

## The matrix

Run a minimal `fn main() { panic!("..."); }` per case under `rustc 1.96.0 (ac68faa20 2026-05-25)` on Windows. The "in isolation" column is verified by `tools/repro/u2014/run.sh` (re-run today; the script reports rc for each case).

| # | Case                                              | In isolation         | In `crates/engine/src/lib.rs` panic context     |
|---|---------------------------------------------------|----------------------|--------------------------------------------------|
| 1 | `panic!("x \u2014 y")` (legacy)                   | REJECTED (post-round verification) | REJECTED (`incorrect unicode escape sequence` at lines 1238:126, 1249:87) |
| 2 | `panic!("x \u{2014} y")` (curly)                  | ACCEPTED             | ACCEPTED (chosen primary form)                  |
| 3 | `panic!("x " + 0xE2 0x80 0x94 + " y")` (raw UTF-8) | ACCEPTED            | ACCEPTED (valid fallback)                       |
| 4 | `let s = "x \u2014 y";` (non-panic binding)       | (not yet run)        | (n/a -- never tested in this context)            |

### How to interpret

- **The legacy 4-digit form `\u2014` rejects in rustc 1.96 even in isolation** (one-liner `fn main() { panic!("...  \u2014 ..."); }` compile fails with `error: incorrect unicode escape sequence`). Row 1 in-isolation was previously thought to be ACCEPTED because the original matrix harness had a case-07 smoking gun (`\uZZZZ` invalid-hex control also reported ACCEPTED); the harness did not gate on rustc exit code or binary existence. The retest on `tools/repro/u2014/case_panic_legacy.rs` rejected cleanly, which is the truth.
- **The curly `\u{2014}` form works everywhere** in normal string literals on rustc 1.96 -- panic!, format!, let bindings, panic-context-specific long panic strings.
- **Raw UTF-8 em-dash bytes (E2 80 94) work everywhere**. Source file is valid UTF-8; the lexer reads 3 bytes as a single U+2014 character.

The most likely upshot: **rustc 1.96 has tightened validation on the legacy 4-digit `\uXXXX` form, regardless of context**. The earlier 'panic-context parse-state sensitivity' hypothesis was speculative; the simpler hypothesis (legacy form rejected universally) fits all observed data points with no remaining anomalies.

## What we changed

`crates/engine/src/lib.rs` `mod eol_invariant` now uses `` `\u{2014}` `` (curly form, ASCII source) for the em-dash in the two `panic!` messages:

```rust
panic!(
    "`{}` is not valid UTF-8: {e}\n                     \
     (typical cause: a Windows-cp1252 byte leaked through \u{2014}                       \
     re-save the file as UTF-8)",
    path.display()
);

panic!(
    "`{}` contains a bare CR at byte offset {} (line {}, 1-based) \u{2014}                          \
     normalize to LF (see .gitattributes / .editorconfig)",
    path.display(),
    i,
    line,
);
```

If you change the form, verify with `cargo build -p voxel-engine` and the `eol_invariant` test before committing.

## Regression hypothesis (best-effort)

Two hypotheses, ranked:

1. **rustc 1.96 tightened validation on the legacy 4-digit `\uXXXX` form generally** in normal `"..."` strings (not panic-context-specific). The simplest model that fits every observed data point. Worth confirming against rustc upstream if a future clang/MSVC-toolchain version changes the behavior, since this is a parser-level decision.

2. ~~(now superseded) rustc 1.96 parse-state sensitivity around legacy 4-digit `\u` inside `panic!` after multi-arg format specifiers.~~ Superseded by hypothesis 1 because hypothesis 1 explains the same data simpler.

If you see the rejection come back on a different rustc version, run `tools/repro/u2014/run.sh` to compare against this matrix; the script will print per-case rc + stderr in real time.
