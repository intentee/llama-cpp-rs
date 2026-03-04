---
name: testing-inference
description: Runs inference verification for the llama-cpp-rs workspace. Use after modifying token decoding, model loading, context creation, sampling, or any code path that affects text generation.
---

# Inference Testing

Verify the llama-cpp-rs workspace produces correct inference output.

## Steps

1. Run clippy and format checks:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets 2>&1 | grep "^warning" | grep -v "llama-cpp-sys-2@" | grep -v "generated\|duplicates"
```

Both must produce zero output.

2. Run unit tests:

```bash
cargo test --workspace
```

All tests must pass.

3. Run the `usage` example which auto-downloads a small model and generates a response:

```bash
cargo run --example usage
```

This uses `unsloth/Qwen3.5-0.8B-GGUF` (cached after first download). Confirm the model loads, generates coherent text, and exits without errors.

4. If changes affect tool-call grammar, chat templates, or streaming, also run:

```bash
cargo run --example tools -- hf-model <repo> <model>
cargo run --example openai_stream -- hf-model <repo> <model>
```

## Expected output

The `usage` example should print a think block followed by a conversational response. Any panic, segfault, or garbled output indicates a regression.
