// Repro case: panic!() with raw UTF-8 em-dash bytes (E2 80 94) inline.
// Compile with `rustc 1.96.0` and observe outcome.
//
// Expected in rustc 1.96: ACCEPTED. Source code is valid UTF-8 and the
// lexer does not interpret the bytes as an escape. The em-dash is baked
// directly into the runtime panic message string.
//
//   $ rustc case_panic_raw.rs -o case_panic_raw.exe
//   $ echo $?   # 0
//
// This file's bytes ARE em-dash bytes (0xE2 0x80 0x94), written as the
// actual UTF-8 character. DO NOT "clean up" them to an escape -- the
// purpose of this case is to verify the raw-bytes fallback form.

fn main() {
    // NOTE: this file's em-dash below is the actual 3-byte UTF-8 sequence
    // (E2 80 94) embedded in the source, NOT a `\u{2014}` escape. The
    // bytes were set via byte-level replace after the Write tool produced
    // curly-escape text instead; see docs/notes/u2014_repro.md for the
    // history. If the em-dash disappears on a future edit, re-run the
    // Python swap in `tools/repro/u2014/README.md`.
    panic!("raw UTF-8 em-dash: — next text");
}
