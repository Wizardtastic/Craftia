// Compiles the GLSL chunk shaders to SPIR-V at build time using glslangValidator
// from the Vulkan SDK. The resulting .spv files land in OUT_DIR and are
// include_bytes!'d by the renderer, so no compiled shaders are committed.
//
// Optimizations over the original sequential build:
//  - Timestamp check: skips recompilation when the .spv is already newer than
//    the source (~0.5-1s saved per build when shaders haven't changed).
//  - Parallel compilation: all 10 shaders compile concurrently via
//    std::thread::scope (~1-2s saved on multi-core machines).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;

fn main() {
    // The shaders live at the workspace root under `shaders/`.
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = PathBuf::from(&manifest)
        .join("..")
        .join("..")
        .join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let glslang = find_glslang_validator();

    let shaders = [
        ("chunk.vert", "chunk.vert.spv"),
        ("chunk.frag", "chunk.frag.spv"),
        ("ui.vert", "ui.vert.spv"),
        ("ui.frag", "ui.frag.spv"),
        ("sky.vert", "sky.vert.spv"),
        ("sky.frag", "sky.frag.spv"),
        ("shadow.vert", "shadow.vert.spv"),
        ("shadow.frag", "shadow.frag.spv"),
        ("post.vert", "post.vert.spv"),
        ("post.frag", "post.frag.spv"),
        ("entity.vert", "entity.vert.spv"),
        ("entity.frag", "entity.frag.spv"),
        ("overlay.vert", "overlay.vert.spv"),
        ("overlay.frag", "overlay.frag.spv"),
        ("panorama.vert", "panorama.vert.spv"),
        ("panorama.frag", "panorama.frag.spv"),
        ("particle.vert", "particle.vert.spv"),
        ("particle.frag", "particle.frag.spv"),
        ("aabb_occlusion.vert", "aabb_occlusion.vert.spv"),
        ("aabb_occlusion.frag", "aabb_occlusion.frag.spv"),
        // Phase 1 — GPU-driven rendering: compute frustum cull + indirect vertex.
        ("chunk_cull.comp", "chunk_cull.comp.spv"),
        ("chunk_indirect.vert", "chunk_indirect.vert.spv"),
        // Phase 2 — GPU compute chunk meshing.
        ("chunk_mesh.comp", "chunk_mesh.comp.spv"),
    ];

    // Register rerun-if-changed for all sources + build.rs itself.
    for (src, _) in &shaders {
        println!("cargo:rerun-if-changed={}", shader_dir.join(src).display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    // Filter to only shaders that need recompilation (source newer than SPV,
    // or SPV missing). This skips the expensive glslangValidator invocation
    // entirely when nothing changed (~0.5-1s per incremental build).
    let needs_rebuild: Vec<(&str, &str)> = shaders
        .iter()
        .copied()
        .filter(|(src, dst)| {
            let src_path = shader_dir.join(src);
            let dst_path = out_dir.join(dst);
            if !dst_path.exists() {
                return true;
            }
            let src_mtime = fs::metadata(&src_path).and_then(|m| m.modified()).ok();
            let dst_mtime = fs::metadata(&dst_path).and_then(|m| m.modified()).ok();
            match (src_mtime, dst_mtime) {
                (Some(s), Some(d)) => s > d,
                _ => true,
            }
        })
        .collect();

    if needs_rebuild.is_empty() {
        return;
    }

    // Compile all out-of-date shaders in parallel. Each glslangValidator
    // invocation is independent, so we spawn one thread per shader.
    let errors: Vec<String> = thread::scope(|s| {
        let handles: Vec<_> = needs_rebuild
            .iter()
            .map(|&(src, dst)| {
                let src_path = shader_dir.join(src);
                let dst_path = out_dir.join(dst);
                let glslang = glslang.clone();
                s.spawn(move || {
                    if !src_path.exists() {
                        return Err(format!("shader source not found: {}", src_path.display()));
                    }
                    let status = Command::new(&glslang)
                        .arg("-V") // compile to SPIR-V (Vulkan target)
                        .arg("-o")
                        .arg(&dst_path)
                        .arg(&src_path)
                        .status()
                        .map_err(|e| format!("failed to run glslangValidator: {e}"))?;
                    if !status.success() {
                        return Err(format!(
                            "glslangValidator failed to compile {}",
                            src_path.display()
                        ));
                    }
                    Ok(())
                })
            })
            .collect();

        handles
            .into_iter()
            .filter_map(|h| h.join().ok().and_then(|r| r.err()))
            .collect()
    });

    if !errors.is_empty() {
        panic!("shader compilation failed:\n{}", errors.join("\n"));
    }
}

fn find_glslang_validator() -> PathBuf {
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let candidate = Path::new(&sdk).join("Bin").join("glslangValidator.exe");
        if candidate.exists() {
            return candidate;
        }
        let candidate2 = Path::new(&sdk).join("Bin").join("glslangValidator");
        if candidate2.exists() {
            return candidate2;
        }
    }
    // PATH fallback. `where` is Windows-only, so use `which` on Unix; the
    // previous Windows-only lookup made Linux builds panic even when
    // glslangValidator was on PATH (e.g. installed via `apt install
    // glslang-tools` to /usr/bin).
    let finder = if cfg!(windows) { "where" } else { "which" };
    if let Ok(out) = Command::new(finder).arg("glslangValidator").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = s.lines().next() {
                return PathBuf::from(line.trim());
            }
        }
    }
    panic!("glslangValidator not found; set VULKAN_SDK or add it to PATH");
}
