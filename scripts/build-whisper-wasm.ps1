# scripts/build-whisper-wasm.ps1
# Skrip kompilasi lokal whisper.cpp C++ ke WebAssembly SIMD128 + Pthreads untuk Pass2WasmGgmlEngine (#2)
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WhisperCppDir = Join-Path $ProjectRoot "target\debug\build\whisper-rs-sys-8ab2dfad9922eabe\out\whisper.cpp"
$OutDir = Join-Path $ProjectRoot "web\vendor\whisper-cpp"

if (-not (Get-Command "emcc" -ErrorAction SilentlyContinue)) {
    Write-Error "Compiler Emscripten (emcc) belum terpasang di PATH. Silakan aktifkan emsdk terlebih dahulu (emsdk_env.ps1)."
    exit 1
}

if (-not (Test-Path $WhisperCppDir)) {
    Write-Error "Direktori source whisper.cpp tidak ditemukan di: $WhisperCppDir"
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutJs = Join-Path $OutDir "whisper.js"

Write-Host "Mengompilasi whisper.cpp ke WASM SIMD128 + Pthreads..." -ForegroundColor Cyan

$Sources = @(
    (Join-Path $WhisperCppDir "ggml.c"),
    (Join-Path $WhisperCppDir "ggml-alloc.c"),
    (Join-Path $WhisperCppDir "ggml-backend.c"),
    (Join-Path $WhisperCppDir "ggml-quants.c"),
    (Join-Path $WhisperCppDir "whisper.cpp")
)

$ExportedFunctions = "['_whisper_init_from_buffer','_whisper_free','_whisper_full_default_params','_whisper_full','_whisper_full_n_segments','_whisper_full_get_segment_text','_malloc','_free']"
$ExportedRuntimeMethods = "['UTF8ToString','stringToUTF8','HEAPU8','HEAPF32','HEAP32']"

& emcc @Sources `
    -I"$WhisperCppDir" `
    -O3 `
    -msimd128 `
    -pthread `
    -sUSE_PTHREADS=1 `
    -sPTHREAD_POOL_SIZE=4 `
    -sINITIAL_MEMORY=268435456 `
    -sALLOW_MEMORY_GROWTH=1 `
    -sMODULARIZE=1 `
    -sEXPORT_ES6=1 `
    -sEXPORT_NAME="createWhisperModule" `
    -sEXPORTED_FUNCTIONS="$ExportedFunctions" `
    -sEXPORTED_RUNTIME_METHODS="$ExportedRuntimeMethods" `
    -o "$OutJs"

Write-Host "SELESAI: Biner whisper.js dan whisper.wasm berhasil dibuat di $OutDir" -ForegroundColor Green
