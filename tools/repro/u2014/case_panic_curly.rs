// Repro case: panic!() with curly-brace unicode escape `\u{2014}` (em-dash).
// Compile with `rustc 1.96.0` and observe outcome.
//
// Expected in rustc 1.96: ACCEPTED in isolation AND in the production
// panic context (this is the form shipped in
// `crates/engine/src/lib.rs` `eol_invariant` panic messages).
//
//   $ rustc case_panic_curly.rs -o case_panic_curly.exe
//   $ echo $?   # 0
//
// Pair with the in-context reproduction: `cargo build -p voxel-engine`
// confirms the form compiles in the real eol_invariant panic strings.

fn main() {
    panic!("curly-brace escape: \u{2014} next text");
}
