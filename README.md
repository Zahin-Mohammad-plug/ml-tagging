# ML Tagger

An AI-powered video tagging system that uses CLIP vision embeddings, automatic speech recognition (ASR), and optical character recognition (OCR) to suggest tags for video content.

## Architecture

```
┌───────────┐     ┌──────────┐     ┌──────────────────────────────────┐
│   React   │────▶│ FastAPI  │────▶│  Celery Workers (4 queues)       │
│    UI     │◀────│   API    │◀────│  sampling → embeddings → asr/ocr │
└───────────┘     └────┬─────┘     │  → fusion                       │
                       │           └──────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │ PostgreSQL +   │
              │   pgvector     │
              └────────────────┘
```

**Pipeline stages:**

1. **Sampling** – extracts frames from a video at a configurable FPS using FFmpeg.
2. **Embeddings** – generates CLIP/SigLIP vector embeddings for each frame.
3. **ASR + OCR** – transcribes speech (faster-whisper) and reads on-screen text (PaddleOCR/EasyOCR).
4. **Fusion** – combines vision, ASR, and OCR signals to produce per-tag confidence scores and tag suggestions.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Zahin-Mohammad-plug/ml-tagging.git
cd ml-tagging

# 2. Copy and edit environment variables
cp .env.example .env

# 3. Start all services
docker compose up --build
```

The UI will be available at **http://localhost:3001** and the API at **http://localhost:9898**.

## Tag Prompts

Tag definitions live in `prompts/` as JSON files. Each tag has a list of natural-language prompts
that CLIP uses for zero-shot classification. A demo cooking-domain prompt file is included at
`prompts/demo_cooking.json`. Create your own prompt file for your domain and point to it with the
`TAG_PROMPTS_PATH` environment variable.

## Project Structure

```
api/          FastAPI backend (REST endpoints, job orchestration)
workers/      Celery workers (sampling, embeddings, asr_ocr, fusion)
ui/           React 18 + TypeScript + Material UI frontend
store/        PostgreSQL migrations
prompts/      Tag prompt definitions (JSON)
models/       Model checkpoints & calibration data
```

## Configuration

All settings can be tuned via environment variables (see `.env.example`) or through the
Settings page in the UI. Key options include:

| Variable | Default | Description |
|---|---|---|
| `VIDEO_MEDIA_PATH` | `/media/videos` | Host path mounted read-only for video access |
| `SAMPLE_FPS` | `0.5` | Frames per second to extract |
| `VISION_MODEL` | `clip-vit-base-patch32` | CLIP model variant |
| `CLIP_MODEL_NAME` | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | HuggingFace model for fusion embeddings |
| `ASR_MODEL` | `whisper-small` | Whisper model size |
| `OCR_ENGINE` | `paddleocr` | OCR backend (`paddleocr` or `easyocr`) |
| `MAX_CONCURRENT_JOBS` | `3` | Parallel processing limit |

## Requirements

- Docker & Docker Compose v2+
- NVIDIA GPU recommended (CUDA support for CLIP and Whisper)
- ~8 GB RAM minimum for the full stack

## Documentation

- [Quick Start](docs/QUICKSTART.md) — get running in 5 minutes
- [Custom Domain Guide](docs/CUSTOM_DOMAIN.md) — write your own tag prompts for any domain
- [Adapter Guide](docs/ADAPTER_GUIDE.md) — integrate with Jellyfin, Plex, or custom media servers

## License

See [LICENSE](LICENSE).
