# Catetin 🎙️

> *"Tolong catetin rapat ini ya!"*  
> A lightweight, 100% offline, driverless meeting transcription and speaker diarization tool for Windows, Linux, macOS, Web/WASM, and Android.

---

## Overview

**Catetin** is a standalone, privacy-first tool inspired by the reverse-engineering analysis of Google AI Edge Foresight. It captures both sides of a meeting (your microphone + remote speaker audio) with **zero virtual audio cable drivers**, transcribes speech locally in **English, Indonesian, and mixed code-switching**, and separates speakers into `[You]` vs `[Speaker 1]`, `[Speaker 2]`.

### Key Highlights
* **Zero Third-Party Drivers:** Uses native Windows WASAPI loopback (`AUDCLNT_STREAMFLAGS_LOOPBACK`) to capture Zoom, Google Meet, Teams, or browser meeting sound directly from the audio hardware without virtual audio cables.
* **Dual Diarization Modes:**
  * **Online Meeting Mode:** Microphone is isolated deterministically as **`[You]`**, while remote participants on loopback are clustered into **`[Speaker 1]`**, **`[Speaker 2]`**, etc.
  * **Full Mic / In-Person Mode:** In a physical conference room, multiple participants sitting around the table speaking into the same laptop/table microphone are acoustically clustered and separated in real time (supporting 2 to 15+ speakers).
* **Acoustic Echo Cancellation (AEC):** Real-time NLMS adaptive filtering prevents loopback audio playing through your speakers from echoing into your microphone transcript.
* **Tiny Binary & Lazy Models:** The compiled executable is $< 20\text{ MB}$. Multilingual Whisper models (~142 MB base) are downloaded on-demand to `%APPDATA%\catetin\models\` on first run, keeping binaries and repository lightweight.
* **Cross-Platform Single Core:** Headless `catetin-core` compiles to Windows (`.exe`), Linux (`catetin`), macOS (Mach-O ARM64), WebAssembly (`wasm32-unknown-unknown`), and Android (Termux / JNI).

---

## Project Structure

```
catetin/
├── .github/
│   └── workflows/
│       ├── test-build.yml       # PR validation on windows-latest
│       └── release.yml          # Automated release packaging & Semantic Release
├── .releaserc.json              # Semantic Release configuration
│
├── crates/
│   ├── catetin-core/            # Platform-agnostic headless engine
│   │   ├── src/
│   │   │   ├── audio/           # WASAPI capture, 16kHz mono resampler, NLMS AEC
│   │   │   ├── diarization/     # VAD segmenter, online cosine speaker clustering
│   │   │   ├── asr/             # SpeechRecognizer trait, multilingual Whisper engine
│   │   │   ├── model/           # Dynamic model manager & cache (%APPDATA%\catetin\models)
│   │   │   ├── pipeline.rs      # Audio -> AEC -> VAD -> Diarization -> TranscriptEvent
│   │   │   └── events.rs        # Structured transcript event types
│   │   └── examples/
│   │       └── record_test.rs   # Audio capture & separation verification
│   │
│   └── catetin-cli/             # Terminal User Interface frontend (catetin executable)
│       └── src/main.rs
│
├── docs/                        # Research and technical blueprints
│   ├── FORESIGHT_FEATURE_ANALYSIS.md
│   ├── WINDOWS_BUILD_BLUEPRINT.md
│   ├── CROSS_PLATFORM_CORE_BLUEPRINT.md
│   └── DEVELOPMENT_PHASES.md
│
├── Dockerfile                   # Multi-stage builder & host file exporter (zero host bloat)
└── Cargo.toml                   # Cargo workspace root
```

---

## Multi-Platform Support

| Platform | Binary Format | Audio Backend | Target Triple |
| :--- | :--- | :--- | :--- |
| **Windows** | `catetin.exe` (PE) | WASAPI Loopback + Direct Mic | `x86_64-pc-windows-msvc` / `gnu` |
| **Linux** | `catetin` (ELF) | ALSA / PulseAudio / PipeWire | `x86_64-unknown-linux-gnu` |
| **macOS** | `catetin` (Mach-O) | CoreAudio Process Tap | `aarch64-apple-darwin` (Apple Silicon) |
| **Web App** | `catetin.wasm` | WebAudio API + WebGPU | `wasm32-unknown-unknown` |
| **Android** | Standalone / JNI | AAudio / OpenSL ES (Termux) | `aarch64-linux-android` |

---

## Building & Testing

### 1. In Docker (Zero Host Bloat)
No local compilers, MinGW, or Rust toolchain needed on your host machine:

```bash
# Build and run tests inside the container
docker build -t catetin .
docker run --rm -it catetin

# Extract compiled executables directly to your host machine's dist/ folder:
# (Produces dist/catetin.exe for Windows and dist/catetin-linux for Linux)
docker build --target export --output dist/ .
```

### 2. GitHub Actions (Automated Cloud Builds & Semantic Release)
Matches the release pipeline:
1. **Pull Requests & Development (`test-build.yml`):** Runs on PRs to `main`, `development`, or push to `development` to verify unit tests and test production compilation.
2. **Push to `main` (`release.yml`):**
   * Automatically builds the native Windows `catetin.exe` on `windows-latest`.
   * Packages it into `dist/catetin-windows-x64.zip`.
   * Runs **Semantic Release** (`.releaserc.json`), generating automated changelogs, tagging versions (`v1.0.0`), and attaching `catetin-windows-x64.zip` directly to the GitHub Release!

---

## Roadmap & Engineering Phases

* [x] **Phase 1: Dual Audio Capture Engine** (WASAPI Mic + Loopback, NLMS AEC, 16 kHz Mono Resampler)
* [x] **Phase 2: Offline Multilingual Speech-to-Text** (`SpeechRecognizer` trait, streaming greedy search, on-demand model downloader)
* [x] **Phase 3: VAD & Speaker Diarization** (Adaptive Energy VAD, Voiceprint Cosine Clustering, Online Meeting & Full Mic In-Person modes)
* [ ] **Phase 4: Full-Width TUI Application** (`catetin-cli` with `ratatui`, live scrolling transcript, Markdown export)
* [ ] **Phase 5: Optional AI Assistant** (Configurable right-pane summary & action items via Groq / Gemini / Ollama)
* [ ] **Phase 6: Cross-Platform Targets** (Web App / WASM with WebGPU + Android Termux / Compose UI)

For complete architectural details, see the [Documentation Index](docs/).
