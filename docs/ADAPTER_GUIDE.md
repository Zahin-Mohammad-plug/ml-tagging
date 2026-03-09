# Media Source Adapter Guide

ML Tagger uses a simple adapter pattern for media ingestion. This guide explains the current architecture and how to extend it to pull video metadata from external systems like Jellyfin, Plex, or a custom CMS.

## Current architecture

Today, videos are ingested via the `/ingest` API endpoint which accepts metadata directly in the request body:

```
POST /ingest
{
  "video_id": "vid001",
  "title": "Pasta Recipe Episode 3",
  "path": "pasta/ep3.mp4",
  "duration": 1824.5,
  "frame_rate": 30.0
}
```

The path is resolved inside the sampler container using the `VIDEO_MEDIA_PATH` volume mount. This is the **local file adapter** approach — simple, no external dependencies.

## Path resolution

The sampler worker converts host paths to container paths using two environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MEDIA_HOST_PREFIX` | *(empty)* | The host-side path prefix to strip |
| `MEDIA_CONTAINER_PREFIX` | `/media/` | The container-side prefix to prepend |
| `VIDEO_MEDIA_PATH` | `/media/videos` | Docker Compose volume mount source |

**Example**: If your videos are at `T:/downloads/cooking-videos/` and you set:

```dotenv
VIDEO_MEDIA_PATH=T:/downloads/cooking-videos
MEDIA_HOST_PREFIX=T:/downloads/cooking-videos/
MEDIA_CONTAINER_PREFIX=/media/
```

Then a path like `T:/downloads/cooking-videos/pasta/ep3.mp4` becomes `/media/pasta/ep3.mp4` inside the container.

If you pass relative paths (e.g. `pasta/ep3.mp4`), the path is used as-is and resolved against the `/media` mount.

## Writing a custom adapter

To integrate with an external media server, you would:

### 1. Create an adapter module

```
api/app/adapters/
├── __init__.py
├── base.py          # Abstract interface
├── local_file.py    # Current default (filesystem)
└── jellyfin.py      # Your custom adapter
```

### 2. Define the interface

```python
# api/app/adapters/base.py
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class VideoMetadata:
    id: str
    title: str
    path: str
    duration: Optional[float] = None
    frame_rate: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

class MediaSourceAdapter(ABC):
    """Abstract interface for fetching video metadata from any source."""

    @abstractmethod
    async def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        """Fetch metadata for a single video."""
        ...

    @abstractmethod
    async def list_videos(self, limit: int = 50, offset: int = 0) -> List[VideoMetadata]:
        """List available videos."""
        ...

    @abstractmethod
    async def get_file_path(self, video_id: str) -> Optional[str]:
        """Return the container-accessible file path for a video."""
        ...
```

### 3. Implement for your media server

```python
# api/app/adapters/jellyfin.py
import httpx
from .base import MediaSourceAdapter, VideoMetadata

class JellyfinAdapter(MediaSourceAdapter):
    def __init__(self, server_url: str, api_key: str):
        self.server_url = server_url
        self.api_key = api_key

    async def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.server_url}/Items/{video_id}",
                headers={"X-Emby-Token": self.api_key},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return VideoMetadata(
                id=data["Id"],
                title=data["Name"],
                path=data.get("Path", ""),
                duration=data.get("RunTimeTicks", 0) / 10_000_000,
            )

    async def list_videos(self, limit=50, offset=0):
        # Jellyfin API call to list items
        ...

    async def get_file_path(self, video_id: str):
        meta = await self.get_video(video_id)
        return meta.path if meta else None
```

### 4. Wire it into the API

In `api/app/main.py`, inject the adapter as a FastAPI dependency:

```python
from app.adapters.jellyfin import JellyfinAdapter

def get_media_adapter():
    return JellyfinAdapter(
        server_url=settings.jellyfin_url,
        api_key=settings.jellyfin_api_key,
    )

@app.post("/ingest")
async def ingest_video(
    video_id: str,
    adapter: MediaSourceAdapter = Depends(get_media_adapter),
    db=Depends(get_db),
):
    meta = await adapter.get_video(video_id)
    # ... create job with metadata
```

## Potential integrations

| Media Server | API Docs | Auth Method |
|---|---|---|
| **Jellyfin** | [API docs](https://api.jellyfin.org/) | API key header |
| **Plex** | [API docs](https://plexapi.dev/) | X-Plex-Token |
| **Emby** | Similar to Jellyfin | API key |
| **Custom CMS** | Your own REST/GraphQL API | Varies |

The adapter pattern keeps the core ML pipeline completely decoupled from media source specifics. The pipeline only needs a file path and optional metadata — everything else is the adapter's responsibility.
