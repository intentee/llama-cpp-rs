use std::path::{Path, PathBuf};

use glob::glob;

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

pub(crate) use debug_log;

pub fn extract_lib_names(out_dir: &Path, build_shared_libs: bool) -> Vec<String> {
    let lib_pattern = if cfg!(windows) {
        "*.lib"
    } else if cfg!(target_os = "macos") {
        if build_shared_libs { "*.dylib" } else { "*.a" }
    } else if build_shared_libs {
        "*.so"
    } else {
        "*.a"
    };

    let libs_dir = out_dir.join("lib*");
    let pattern = libs_dir.join(lib_pattern);
    debug_log!("Extract libs {}", pattern.display());

    let pattern_str = pattern.to_string_lossy();
    let mut lib_names: Vec<String> = Vec::new();

    let Ok(entries) = glob(&pattern_str) else {
        println!("cargo:warning=failed to glob library pattern: {pattern_str}");

        return lib_names;
    };

    for entry in entries {
        match entry {
            Ok(path) => {
                if let Some(lib_name) = extract_single_lib_name(&path) {
                    lib_names.push(lib_name);
                }
            }
            Err(error) => println!("cargo:warning=glob error: {error}"),
        }
    }

    lib_names
}

fn extract_single_lib_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;

    if let Some(stripped) = stem.strip_prefix("lib") {
        Some(stripped.to_string())
    } else {
        if path.extension() == Some(std::ffi::OsStr::new("a"))
            && let Some(parent) = path.parent()
        {
            let renamed_path = parent.join(format!("lib{stem}.a"));

            if let Err(error) = std::fs::rename(path, &renamed_path) {
                println!(
                    "cargo:warning=failed to rename {} to {}: {error}",
                    path.display(),
                    renamed_path.display()
                );
            }
        }

        Some(stem.to_string())
    }
}

pub fn extract_lib_assets(out_dir: &Path) -> Vec<PathBuf> {
    let shared_lib_pattern = if cfg!(windows) {
        "*.dll"
    } else if cfg!(target_os = "macos") {
        "*.dylib"
    } else {
        "*.so"
    };

    let shared_libs_dir = if cfg!(windows) { "bin" } else { "lib" };
    let libs_dir = out_dir.join(shared_libs_dir);
    let pattern = libs_dir.join(shared_lib_pattern);
    debug_log!("Extract lib assets {}", pattern.display());

    let pattern_str = pattern.to_string_lossy();
    let mut files = Vec::new();

    let Ok(entries) = glob(&pattern_str) else {
        println!("cargo:warning=failed to glob shared lib pattern: {pattern_str}");

        return files;
    };

    for entry in entries {
        match entry {
            Ok(path) => files.push(path),
            Err(error) => eprintln!("cargo:warning=glob error: {error}"),
        }
    }

    files
}
