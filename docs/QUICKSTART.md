# Quick Start

Get ML Tagger running locally with your own videos in under 5 minutes.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Docker Compose v2+)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (recommended, but CPU works)
- A folder of video files you want to tag

## 1. Clone and configure

```bash
git clone https://github.com/Zahin-Mohammad-plug/ml-tagging.git
cd ml-tagging
cp .env.example .env
```

Edit `.env` and set your video directory:

```dotenv
# Point to the folder containing your videos
VIDEO_MEDIA_PATH=T:/downloads/cooking-videos
```

The sampler container mounts this folder read-only at `/media` inside Docker.

## 2. Start the stack

```bash
docker compose up --build
```

This starts 7 services:

| Service | Port | Purpose |
|---|---|---|
| **UI** | [localhost:3001](http://localhost:3001) | React frontend |
| **API** | [localhost:9898](http://localhost:9898) | FastAPI backend |
| **PostgreSQL** | 5432 | Database + pgvector |
| **Redis** | 7200 | Celery message broker |
| **sampler-worker** | — | Frame extraction (FFmpeg) |
| **embeddings-worker** | — | CLIP vision embeddings |
| **asr-ocr-worker** | — | Whisper + PaddleOCR |
| **fusion-worker** | — | Multi-modal scoring |

Wait for the health checks to pass (about 30-60 seconds on first boot while models download).

## 3. Process a video

Open the UI at **http://localhost:3001**.

1. Go to **Process Video**.
2. Enter a video file path relative to your media folder (e.g. `ep1.mp4` or `subfolder/ep1.mp4`) or use the video browser.
3. Click **Process** — the pipeline runs: sampling → embeddings → ASR/OCR → fusion.
4. When complete, go to **Review** to see tag suggestions with confidence scores.
5. **Approve** or **Reject** each suggestion.

## 4. Check the API directly

```bash
# Health check
curl http://localhost:9898/health

# Ingest a video
curl -X POST http://localhost:9898/ingest \
  -H "Content-Type: application/json" \
  -d '{"video_id": "vid001", "title": "Pasta Recipe", "path": "pasta-recipe.mp4"}'

# List jobs
curl http://localhost:9898/jobs

# List suggestions
curl http://localhost:9898/suggestions
```

## 5. Customize tags

The system uses **tag prompts** for CLIP zero-shot classification. The default set is for cooking (`prompts/demo_cooking.json`). To use your own domain, see [CUSTOM_DOMAIN.md](CUSTOM_DOMAIN.md).

## Troubleshooting

| Issue | Fix |
|---|---|
| `No GPU detected` in embeddings worker | Install NVIDIA Container Toolkit, or set `VISION_DEVICE=cpu` in `.env` (slower) |
| Frames not extracted | Check `VIDEO_MEDIA_PATH` points to the right folder and videos are readable |
| Low confidence scores | Tune thresholds in **Settings** page, or write better prompts (see [CUSTOM_DOMAIN.md](CUSTOM_DOMAIN.md)) |
| Port conflict | Change `UI_PORT` or the API port mapping in `docker-compose.yml` |
