# llama-cpp-rs

Rust bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp). Minimal, close to raw bindings, kept up to date with upstream.

## Quick start

```bash
git clone --recursive https://github.com/utilityai/llama-cpp-rs
cd llama-cpp-rs
cargo run --release --example usage -- "Hello, how are you?"
```

Add `--features cuda` for GPU acceleration.

## Crates

| Crate | Description |
|---|---|
| `llama-cpp-2` | Safe Rust API |
| `llama-cpp-sys-2` | Raw FFI bindings |
| `llama-cpp-build` | Build system |
