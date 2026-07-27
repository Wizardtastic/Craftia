# `tools/repro/u2014/` — rustc 1.96 unicode-escape repro artifacts

These `.rs` files plus `run.sh` / `run.bat` are the executable companion to
[`docs/notes/u2014_repro.md`](../../../docs/notes/u2014_repro.md). The
markdown note explains the regression; these files let you re-verify it.

## Files

| File                     | What it does                                              |
|--------------------------|-----------------------------------------------------------|
| `case_panic_legacy.rs`   | `panic!()` with legacy 4-digit escape `\u2014`           |
| `case_panic_curly.rs`    | `panic!()` with curly escape `\u{2014}` (chosen form)    |
| `case_panic_raw.rs`      | `panic!()` with raw UTF-8 em-dash bytes                  |
| `run.sh`                 | POSIX shell wrapper; compiles each case, reports rc      |
| `run.bat`                | Windows batch sibling                                    |

## Re-running

```bash
# POSIX (Linux / macOS / git-bash on Windows)
tools/repro/u2014/run.sh

# Windows cmd
tools\repro\u2014\run.bat
```

Expected outcome on **rustc 1.96.0 (ac68faa20 2026-05-25)**: all three
cases compile cleanly (rc=0). That confirms the **in-isolation** column
of the matrix.

The **in production panic context** column requires building the real
crate:

```bash
cargo build -p voxel-engine
```

After the round where we settled on `` `\u{2014}` `` as the chosen form
for `crates/engine/src/lib.rs` `mod eol_invariant` panic messages, that
build succeeds with no `incorrect unicode escape sequence` errors.

If you want to independently RE-trigger the rejected legacy form for
documentation purposes, hand-edit `crates/engine/src/lib.rs` to use
`` `\u2014` `` (4-digit legacy) at the panic sites and run `cargo build
-p voxel-engine`; the rejection will re-appear. Capture the stderr to
`recorded_stderr.txt` so the rejection becomes a real artifact.
