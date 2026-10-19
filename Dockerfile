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
    CFLAGS_x86_64_pc_windows_gnu="-D_WIN32_WINNT=0x0601 -mstackrealign -O3" \
    CXXFLAGS_x86_64_pc_windows_gnu="-D_WIN32_WINNT=0x0601 -mstackrealign -O3" \
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-g++-posix \
    CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS="-C link-arg=-static -C link-arg=-static-libgcc -C link-arg=-static-libstdc++ -C link-arg=-Wl,--stack,16777216"

WORKDIR /workspace

# Cache dependency layer (pre-fetch crates and apply whisper.cpp MinGW SIMD/static patches once)
COPY Cargo.toml Cargo.lock* ./
COPY crates/catetin-core/Cargo.toml ./crates/catetin-core/
COPY crates/catetin-cli/Cargo.toml ./crates/catetin-cli/
COPY crates/catetin-wasm/Cargo.toml ./crates/catetin-wasm/

RUN mkdir -p crates/catetin-core/src crates/catetin-cli/src crates/catetin-wasm/src \
    && echo "" > crates/catetin-core/src/lib.rs \
    && echo "fn main() {}" > crates/catetin-cli/src/main.rs \
    && echo "" > crates/catetin-wasm/src/lib.rs \
    && cargo build --release -p catetin-cli \
    && CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-g++-posix \
       RUSTFLAGS="-C target-feature=+avx,+avx2,+fma,+f16c -C link-arg=-static -C link-arg=-static-libgcc -C link-arg=-static-libstdc++ -C link-arg=-Wl,--stack,16777216" \
       cargo build --target x86_64-pc-windows-gnu --release -p catetin-cli \
    && cargo build --target wasm32-unknown-unknown --release -p catetin-wasm \
    && rm -rf crates/*/src target/release/deps/catetin* target/release/deps/libcatetin* \
       target/x86_64-pc-windows-gnu/release/deps/catetin* target/x86_64-pc-windows-gnu/release/deps/libcatetin* \
       target/wasm32-unknown-unknown/release/deps/catetin* target/wasm32-unknown-unknown/release/deps/libcatetin*

# Copy workspace source files
COPY crates ./crates

# 1. Run unit tests (offline using cached registry)
RUN cargo test --offline -p catetin-core --no-default-features

# 2. Build native Linux binary (offline using pre-compiled dependencies)
RUN cargo build --offline --release -p catetin-cli

# 4. Cross-compile native Windows executable with in-process Whisper (catetin.exe)
RUN CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-g++-posix \
    RUSTFLAGS="-C target-feature=+avx,+avx2,+fma,+f16c -C link-arg=-static -C link-arg=-static-libgcc -C link-arg=-static-libstdc++ -C link-arg=-Wl,--stack,16777216" \
    cargo build --offline --target x86_64-pc-windows-gnu --release -p catetin-cli

# 5. Compile WebAssembly binary (catetin_wasm.wasm) for web browsers and frontend UI
RUN cargo build --offline --target wasm32-unknown-unknown --release -p catetin-wasm

# ============================================================
# Stage 2: Host File Exporter
# Usage: docker build --target export --output dist/ .
# Exports standalone catetin.exe (Windows), catetin-linux, and catetin_wasm.wasm directly to your host!
# ============================================================
FROM scratch AS export
COPY --from=builder /workspace/target/x86_64-pc-windows-gnu/release/catetin.exe /catetin.exe
COPY --from=builder /workspace/target/release/catetin /catetin
COPY --from=builder /workspace/target/wasm32-unknown-unknown/release/catetin_wasm.wasm /catetin.wasm
COPY crates/catetin-core/native/win-x64/sherpa-onnx-c-api.dll /sherpa-onnx-c-api.dll
COPY crates/catetin-core/native/win-x64/onnxruntime.dll /onnxruntime.dll
COPY web/ /web/

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
