use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target = env::var("TARGET").expect("TARGET");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let vendor_root = manifest_dir.join("vendor/iree_tokenizer_src");
    let source_root = manifest_dir;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=sources/base_sources.txt");
    println!("cargo:rerun-if-changed=sources/tokenizer_sources.txt");
    println!("cargo:rerun-if-changed=vendor/iree_tokenizer_src");

    let mut build = cc::Build::new();
    build
        .include(&vendor_root)
        .cargo_metadata(false)
        .warnings(false)
        .flag_if_supported("-std=gnu11")
        .define("IREE_ALLOCATOR_SYSTEM_CTL", "iree_allocator_libc_ctl")
        .define("IREE_STATUS_MODE", "2")
        .define("IREE_TRACING_MODE", "0")
        .define("IREE_THREADING_ENABLE", "0")
        .define("IREE_FILE_IO_ENABLE", "0");

    for manifest in ["sources/base_sources.txt", "sources/tokenizer_sources.txt"] {
        for source in read_manifest(&source_root.join(manifest)) {
            build.file(source_root.join(source));
        }
    }

    build.compile("iree_tokenizer_bundle");

    let archive = out_dir.join("libiree_tokenizer_bundle.a");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", archive.display());
    } else {
        println!("cargo:rustc-link-arg=-Wl,--whole-archive");
        println!("cargo:rustc-link-lib=static=iree_tokenizer_bundle");
        println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");
    }
}

fn read_manifest(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect()
}
