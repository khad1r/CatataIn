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

# Cache dependency layer
COPY Cargo.toml Cargo.lock* ./
COPY crates/catetin-core/Cargo.toml ./crates/catetin-core/
COPY crates/catetin-cli/Cargo.toml ./crates/catetin-cli/
COPY crates/catetin-wasm/Cargo.toml ./crates/catetin-wasm/

# Copy workspace source files
COPY crates ./crates

# 1. Run unit tests
RUN cargo test -p catetin-core --no-default-features

# 2. Build native Linux binary
RUN cargo build --release -p catetin-cli

# 3. Patch whisper.cpp/ggml CMakeLists on MinGW to keep 'lib' prefix for static libraries (rustc expects libggml.a),
# enable AVX/AVX2/FMA/F16C SIMD instructions (disabled by default when CMAKE_CROSSCOMPILING=TRUE),
# and force 100% static C++ runtime linkage into the single .exe binary
RUN grep -rn "CMAKE_STATIC_LIBRARY_PREFIX" /usr/local/cargo/registry/src/ || true \
    && find /usr/local/cargo/registry/src/ -name CMakeLists.txt -exec sed -i 's/set(CMAKE_STATIC_LIBRARY_PREFIX "")//g' {} + \
    && find /usr/local/cargo/registry/src/ -path "*whisper.cpp*CMakeLists.txt" -exec sed -i '1i add_compile_options(-O3 -mavx -mavx2 -mfma -mf16c -mstackrealign)' {} + \
    && find /usr/local/cargo/registry/src/ -name build.rs -exec sed -i 's/target.contains("msvc")/target.contains("windows")/g' {} + \
    && mv /usr/lib/gcc/x86_64-w64-mingw32/12-posix/libstdc++.dll.a /usr/lib/gcc/x86_64-w64-mingw32/12-posix/libstdc++.dll.a.bak 2>/dev/null || true

# 4. Cross-compile native Windows executable with in-process Whisper (catetin.exe)
RUN CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-g++-posix \
    RUSTFLAGS="-C target-feature=+avx,+avx2,+fma,+f16c -C link-arg=-static -C link-arg=-static-libgcc -C link-arg=-static-libstdc++ -C link-arg=-Wl,--stack,16777216" \
    cargo build --target x86_64-pc-windows-gnu --release -p catetin-cli

# 5. Compile WebAssembly binary (catetin_wasm.wasm) for web browsers and frontend UI
RUN cargo build --target wasm32-unknown-unknown --release -p catetin-wasm

# ============================================================
# Stage 2: Host File Exporter
# Usage: docker build --target export --output dist/ .
# Exports standalone catetin.exe (Windows), catetin-linux, and catetin_wasm.wasm directly to your host!
# ============================================================
FROM scratch AS export
COPY --from=builder /workspace/target/x86_64-pc-windows-gnu/release/catetin.exe /catetin.exe
COPY --from=builder /workspace/target/release/catetin /catetin
COPY --from=builder /workspace/target/wasm32-unknown-unknown/release/catetin_wasm.wasm /catetin.wasm

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
