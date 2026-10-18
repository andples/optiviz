# optiviz tasks. `just` lists them.

wasm_out := "web/src/wasm"

default:
    @just --list

# Rust tests (core with and without threads)
test:
    cargo test --workspace --release --exclude optiviz-gpu-test
    cargo test -p optiviz-core --release --features parallel

# Kernel tests: WGSL shaders run through native wgpu against the Rust reference
test-gpu:
    cargo test -p optiviz-gpu-test --release -- --nocapture

lint:
    cargo fmt --check
    cargo clippy --workspace --all-targets -- -D warnings
    cd web && pnpm check

# Both WebAssembly builds: single-threaded (SIMD) and threaded (SIMD + atomics + rayon)
wasm: wasm-st wasm-threads

wasm-st:
    RUSTFLAGS="-C target-feature=+simd128" \
      cargo build -p optiviz-wasm --release --target wasm32-unknown-unknown --target-dir target/wasm-st
    wasm-bindgen --target web --out-dir {{wasm_out}}/pkg-st target/wasm-st/wasm32-unknown-unknown/release/optiviz_wasm.wasm

# std is rebuilt with atomics (-Z build-std); RUSTC_BOOTSTRAP=1 allows that on stable (docs/adr/0002-toolchain.md)
wasm-threads:
    RUSTC_BOOTSTRAP=1 RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
      -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296 \
      -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
      cargo build -p optiviz-wasm --release --features threads --target wasm32-unknown-unknown --target-dir target/wasm-threads \
      -Z build-std=panic_abort,std
    wasm-bindgen --target web --out-dir {{wasm_out}}/pkg-threads target/wasm-threads/wasm32-unknown-unknown/release/optiviz_wasm.wasm
    # wasm-bindgen-rayon's worker helper imports the package by directory ('../../..'); Vite can't resolve a
    # relative directory import, so point it at the module file
    sed -i "s#import('../../..')#import('../../../optiviz_wasm.js')#" {{wasm_out}}/pkg-threads/snippets/wasm-bindgen-rayon-*/src/workerHelpers.js

# Native benchmarks (JSON on stdout)
bench:
    cargo run -p optiviz-bench --release

# Dev server with COOP/COEP headers
dev: wasm
    cd web && pnpm dev

build: wasm
    cd web && pnpm build

# Browser tests (Playwright). CHROMIUM=/path uses a local Chromium instead of Playwright's.
e2e: build
    cd web && pnpm e2e
