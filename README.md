# Catetin 🎙️

> *"Tolong catetin rapat ini ya!"*  
> A lightweight, 100% offline, driverless meeting transcription and speaker diarization tool for Windows, Web/WASM, and Android.

---

## Overview

**Catetin** is designed as a standalone, privacy-first tool inspired by the reverse-engineering analysis of Google AI Edge Foresight. It captures both sides of a meeting (your microphone + remote speaker audio) with **zero virtual audio cable drivers**, transcribes speech locally in **English, Indonesian, and mixed code-switching**, and separates speakers into `[You]` vs `[Speaker 1]`, `[Speaker 2]`.

### Key Highlights
* **Zero Third-Party Drivers:** Uses native Windows WASAPI loopback (`AUDCLNT_STREAMFLAGS_LOOPBACK`) to capture Zoom, Google Meet, Teams, or browser meeting sound directly from the audio hardware.
* **Deterministic Speaker Diarization:** Your microphone is hardware-isolated as **`[You]`**. Remote participants are distinguished using acoustic voiceprint clustering.
* **Acoustic Echo Cancellation (AEC):** Real-time NLMS adaptive filtering prevents loopback audio playing through your speakers from echoing into your microphone transcript.
* **Tiny Binary & Lazy Models:** The executable is $< 20\text{ MB}$. Whisper models (~142 MB base) are downloaded on-demand to `%APPDATA%\catetin\models\` on first run, never bloating the repository.
* **Cross-Platform Single Core:** Headless `catetin-core` compiles to native Windows CLI/TUI, WebAssembly (`wasm32-unknown-unknown`), and Android (Termux / JNI).

---

## Project Structure

```
catetin/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated Windows .exe & WASM builds via GitHub Actions
├── crates/
│   ├── catetin-core/            # Platform-agnostic headless engine
│   │   ├── src/
│   │   │   ├── audio/           # WASAPI capture, 16kHz mono resampler, NLMS AEC
│   │   │   ├── diarization/     # Silero VAD segmenter, online cosine speaker clustering
│   │   │   ├── asr/             # SpeechRecognizer trait, multilingual Whisper engine
│   │   │   ├── model/           # Dynamic model manager & cache (%APPDATA%\catetin\models)
│   │   │   ├── pipeline.rs      # Audio -> AEC -> VAD -> Diarization -> TranscriptEvent
│   │   │   └── events.rs        # Structured transcript event types
│   │   └── examples/
│   │       └── record_test.rs   # Audio capture & separation test
│   │
│   └── catetin-cli/             # Terminal User Interface frontend (catetin.exe)
│       └── src/main.rs
│
├── docs/                        # Research and technical blueprints
│   ├── FORESIGHT_FEATURE_ANALYSIS.md
│   ├── WINDOWS_BUILD_BLUEPRINT.md
│   ├── CROSS_PLATFORM_CORE_BLUEPRINT.md
│   └── DEVELOPMENT_PHASES.md
│
├── Dockerfile                   # Hardened slim Docker build container (zero host bloat)
└── Cargo.toml                   # Cargo workspace root
```

---

## Building & Testing

### 1. In Docker (Zero Host Bloat)
No local compilers, MinGW, or Rust toolchain needed on your host machine:

```bash
# Build the isolated container
docker build -t catetin .

# Run the test suite inside the container
docker run --rm catetin
```

### 2. GitHub Actions (Automated Cloud Builds & Semantic Release)
Matches the release pipeline:
1. **Pull Requests (`test-build.yml`):** Runs on PRs to `main`, `master`, and `development` to verify unit tests and test production compilation.
2. **Push to `main` / `master` (`release.yml`):**
   * Automatically builds the native Windows `catetin.exe` on `windows-latest`.
   * Packages it into `dist/catetin-windows-x64.zip`.
   * Runs **Semantic Release** (`.releaserc.json`), generating automated changelogs, tagging versions (`v1.0.0`), and attaching `catetin-windows-x64.zip` directly to the GitHub Release!

---

## Roadmap & Engineering Phases

* [x] **Phase 1: Dual Audio Capture Engine** (WASAPI Mic + Loopback, NLMS AEC, 16 kHz Mono Resampler)
* [x] **Phase 2: Offline Multilingual Speech-to-Text** (`SpeechRecognizer` trait, streaming greedy search, on-demand model downloader)
* [x] **Phase 3: VAD & Speaker Diarization** (Adaptive Energy VAD, Voiceprint Cosine Clustering, `[You]` vs `[Speaker N]`)
* [ ] **Phase 4: Full-Width TUI Application** (`catetin-cli` with `ratatui`, live scrolling transcript, Markdown export)
* [ ] **Phase 5: Optional AI Assistant** (Configurable right-pane summary & action items via Groq / Gemini / Ollama)
* [ ] **Phase 6: Cross-Platform Targets** (Web App / WASM with WebGPU + Android Termux / Compose UI)

For complete architectural details, see the [Documentation Index](docs/).
