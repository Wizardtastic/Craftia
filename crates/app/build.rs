//! Build script for `voxel-app`. Captures build-time metadata so the
//! runtime log can identify "what run / what commit / what target is
//! this?" at a glance. All values are exposed to the binary through
//! `cargo:rustc-env=VOXEL_*=...` and read in `main.rs` via `env!()`.
//!
//! Every step here is best-effort: a missing `git`, a release tarball
//! build, or a non-`rustc` toolchain is OK — we fall back to "unknown"
//! so the binary still builds.

use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn main() {
    // Rerun the build script when HEAD shifts (typical dev loop) but be
    // tolerant of build environments without a git directory (CI
    // source tarballs, `cargo publish`).
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
    println!("cargo:rerun-if-changed=../../.git/index");

    let sha = git_output(&["rev-parse", "--short", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = git_output(&["status", "--porcelain"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let build_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Cargo always sets TARGET; fall back to "unknown" only if the build
    // environment somehow doesn't, which would be a cargo bug.
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());

    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=VOXEL_GIT_SHA={}", sha);
    println!("cargo:rustc-env=VOXEL_GIT_DIRTY={}", dirty);
    println!("cargo:rustc-env=VOXEL_BUILD_UNIX={}", build_unix);
    println!("cargo:rustc-env=VOXEL_TARGET_TRIPLE={}", target);
    println!("cargo:rustc-env=VOXEL_RUSTC_VERSION={}", rustc);
}
