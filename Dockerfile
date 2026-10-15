# Multi-stage Docker container for building & testing Catetin with zero host bloat
FROM rust:1.85-bookworm AS builder

# Install audio dependencies, clang, and cross-compilation tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    clang \
    libclang-dev \
    llvm \
    ninja-build \
    libasound2-dev \
    pkg-config \
    mingw-w64 \
    && rm -rf /var/lib/apt/lists/*

# Add targets for Windows cross-compilation and WebAssembly
RUN rustup target add x86_64-pc-windows-gnu wasm32-unknown-unknown

WORKDIR /workspace

# Copy workspace manifests
COPY Cargo.toml ./
COPY crates/catetin-core/Cargo.toml ./crates/catetin-core/
COPY crates/catetin-cli/Cargo.toml ./crates/catetin-cli/

# Copy source tree
COPY crates ./crates

# Build & test by default
CMD ["cargo", "test", "--workspace", "--no-default-features"]
