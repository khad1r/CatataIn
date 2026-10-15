# ============================================================
# Stage 1: Builder (Compiles Linux & Windows x64 .exe)
# ============================================================
FROM rust:1-slim-bookworm AS builder

# Apply security upgrades and install minimal build tools + mingw-w64 cross compiler
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        cmake \
        make \
        clang \
        libclang-dev \
        llvm \
        ninja-build \
        libasound2-dev \
        pkg-config \
        mingw-w64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Windows cross-compilation & WebAssembly targets
RUN rustup component add rustfmt \
    && rustup target add x86_64-pc-windows-gnu wasm32-unknown-unknown

# Configure MinGW to use POSIX thread model (enables C++11 std::mutex for whisper.cpp / ggml)
RUN update-alternatives --set x86_64-w64-mingw32-gcc /usr/bin/x86_64-w64-mingw32-gcc-posix \
    && update-alternatives --set x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix

ENV CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc-posix \
    CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++-posix \
    CFLAGS_x86_64_pc_windows_gnu="-D_WIN32_WINNT=0x0A00" \
    CXXFLAGS_x86_64_pc_windows_gnu="-D_WIN32_WINNT=0x0A00" \
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc-posix

WORKDIR /workspace

# Cache dependency layer
COPY Cargo.toml Cargo.lock* ./
COPY crates/catetin-core/Cargo.toml ./crates/catetin-core/
COPY crates/catetin-cli/Cargo.toml ./crates/catetin-cli/

# Copy workspace source files
COPY crates ./crates

# 1. Run unit tests
RUN cargo test --workspace --no-default-features

# 2. Build native Linux binary
RUN cargo build --release -p catetin-cli

# 3. Cross-compile native Windows executable (catetin.exe)
RUN cargo build --target x86_64-pc-windows-gnu --release -p catetin-cli --no-default-features

# ============================================================
# Stage 2: Host File Exporter
# Usage: docker build --target export --output dist/ .
# Exports both catetin.exe (Windows) and catetin-linux directly to your host!
# ============================================================
FROM scratch AS export
COPY --from=builder /workspace/target/x86_64-pc-windows-gnu/release/catetin.exe /catetin.exe
COPY --from=builder /workspace/target/release/catetin /catetin-linux

# ============================================================
# Stage 3: Minimal Runtime Container (Self-contained, ~35 MB)
# Default stage for `docker run --rm -it catetin`
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
