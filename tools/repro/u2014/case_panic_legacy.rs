// Repro case: panic!() with legacy 4-digit unicode escape `\u2014` (em-dash).
// Compile with `rustc 1.96.0` and observe outcome.
//
// Expected in rustc 1.96: ACCEPTED in isolation (case 1 of the matrix).
//   $ rustc case_panic_legacy.rs -o case_panic_legacy.exe
//   $ echo $?   # 0
//
// Per the matrix in docs/notes/u2014_repro.md, the legacy form REJECTS in
// the specific panic-string context used by `crates/engine/src/lib.rs`
// `eol_invariant::all_crates_rs_files_are_pure_lf_utf8` -- reproduce THAT
// by running cargo build against the real source, not this isolated case.
//
// This file lives at tools/repro/u2014/case_panic_legacy.rs so the
// matrix is re-runnable; it's not used as a build dependency.

fn main() {
    panic!("legacy 4-digit escape: \u2014 next text");
}
