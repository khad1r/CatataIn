# Secure, minimal Rust build container for Catetin
FROM rust:1-slim-bookworm AS builder

# 1. Apply latest security patches to resolve base image CVEs
# 2. Install only essential build tools (Audio headers, Clang/LLVM, CMake, Ninja)
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        cmake \
        clang \
        libclang-dev \
        llvm \
        ninja-build \
        libasound2-dev \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add WebAssembly target
RUN rustup target add wasm32-unknown-unknown

WORKDIR /workspace

# Copy manifests first for efficient Docker layer caching
COPY Cargo.toml ./
COPY crates/catetin-core/Cargo.toml ./crates/catetin-core/
COPY crates/catetin-cli/Cargo.toml ./crates/catetin-cli/

# Copy workspace source files
COPY crates ./crates

# Default: execute core tests in an isolated, secure environment
CMD ["cargo", "test", "--workspace", "--no-default-features"]
