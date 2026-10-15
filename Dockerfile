# ============================================================
# Stage 1: Builder (Compiles tests and release executable)
# ============================================================
FROM rust:1-slim-bookworm AS builder

# Apply security upgrades and install minimal build tools
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

RUN rustup target add wasm32-unknown-unknown

WORKDIR /workspace

# Cache dependency layer
COPY Cargo.toml Cargo.lock* ./
COPY crates/catetin-core/Cargo.toml ./crates/catetin-core/
COPY crates/catetin-cli/Cargo.toml ./crates/catetin-cli/

# Copy workspace source files
COPY crates ./crates

# Run test suite & build release binary
RUN cargo test --workspace --no-default-features \
    && cargo build --release -p catetin-cli

# ============================================================
# Stage 2: Minimal Runtime Container (Self-contained, ~35 MB)
# ============================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libasound2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /workspace/target/release/catetin /usr/local/bin/catetin

ENTRYPOINT ["catetin"]

# ============================================================
# Stage 3: Host File Exporter
# Usage: docker build --target export --output dist/ .
# Extracts the compiled executable directly to your host machine!
# ============================================================
FROM scratch AS export
COPY --from=builder /workspace/target/release/catetin /catetin
