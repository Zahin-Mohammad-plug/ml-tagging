# Custom Domain Guide

ML Tagger is domain-agnostic — it works with any set of visual/audio concepts you define. This guide shows how to create your own tag prompts and tune the system for a new domain.

## How CLIP zero-shot classification works

CLIP embeds both images and text into the same vector space. For each frame of a video, the system computes similarity between the frame embedding and a set of **text prompts** you provide. If a prompt like `"a person chopping vegetables on a cutting board"` is highly similar to a frame, the tag `"Chopping"` gets a high confidence score.

More prompts per tag → better coverage of visual variations → more reliable tagging.

## Step 1: Create a prompts file

Create a JSON file in the `prompts/` directory:

```json
{
  "Tag Name": [
    "a descriptive sentence of what this looks like in a video frame",
    "another visual description capturing a different angle or variation",
    "a third prompt covering edge cases"
  ],
  "Another Tag": [
    "description one",
    "description two"
  ]
}
```

### Example: Security cameras

```json
{
  "Person Detected": [
    "a person walking through a hallway captured by a security camera",
    "a human figure visible in surveillance footage",
    "someone entering a room seen from a ceiling-mounted camera"
  ],
  "Vehicle": [
    "a car in a parking lot viewed from a security camera",
    "a vehicle driving past captured on CCTV footage"
  ],
  "Package Delivery": [
    "a delivery person leaving a package at a door",
    "a box or parcel being placed on a doorstep"
  ]
}
```

### Tips for writing good prompts

| Do | Don't |
|---|---|
| Describe what the **frame** looks like | Use abstract concepts ("happiness", "danger") |
| Include 3-5 prompts per tag for variety | Use a single very specific prompt |
| Mention visual context ("in a pan", "on a cutting board") | Write one-word prompts ("chopping") |
| Cover different angles/lighting/zoom levels | Assume perfect conditions only |
| Use natural language sentences | Use keyword lists |

CLIP responds best to prompts structured like image captions: *"a photo of [thing happening]"* or *"[person/object] doing [action] in [setting]"*.

## Step 2: Point the system to your prompts

Set the `TAG_PROMPTS_PATH` environment variable in your `.env`:

```dotenv
TAG_PROMPTS_PATH=/app/prompts/my_domain.json
```

Or place your file as `prompts/demo_cooking.json` (the default path) to avoid config changes.

The tags are synced to the database on API startup. You can also trigger a re-sync from the **Tags** page in the UI.

## Step 3: Tune thresholds

After processing a few videos, you'll want to adjust confidence thresholds:

| Setting | Default | What it controls |
|---|---|---|
| `DEFAULT_REVIEW_THRESHOLD` | `0.3` | Minimum confidence to create a suggestion (for human review) |
| `DEFAULT_AUTO_THRESHOLD` | `0.8` | Confidence above which tags are auto-approved |
| `VISION_WEIGHT` | `0.7` | How much CLIP vision contributes to final score |
| `ASR_WEIGHT` | `0.2` | How much speech transcription contributes |
| `OCR_WEIGHT` | `0.1` | How much on-screen text contributes |
| `MIN_AGREEMENT_FRAMES` | `3` | Minimum frames that must agree for a tag to be suggested |

These can be changed in `.env`, via the `/settings` API, or in the **Settings** page of the UI.

### Per-tag thresholds

Each tag can have its own `review_threshold` and `auto_threshold` in the database. Edit these on the **Tags** page in the UI. For example, a visually distinctive tag like `"Grilling"` might work well at 0.25, while an ambiguous one like `"Seasoning"` might need 0.45.

## Step 4: Fine-tuning (optional)

If zero-shot accuracy isn't sufficient, you can fine-tune a CLIP model on your labeled data:

1. Organize labeled frames into folders: `trainData/<Tag Name>/frame001.jpg`
2. Run `scripts/prepare_dataset.py` to build the training manifest
3. Run `scripts/train_model.py` to fine-tune
4. Point the worker to the new checkpoint via `CALIBRATION_MODEL_PATH`

This is optional — most domains work well with carefully written prompts alone.

## Domain ideas

| Domain | Example Tags | CLIP accuracy |
|---|---|---|
| **Cooking** | Chopping, Grilling, Plating, Baking | High |
| **Security** | Person, Vehicle, Package, Animal | High |
| **Sports** | Goal, Foul, Celebration, Replay | Medium |
| **Nature/Wildlife** | Bird, Deer, Sunset, Storm | High |
| **Fitness** | Squat, Pushup, Running, Stretching | Medium-High |
| **Retail** | Customer, Empty Shelf, Checkout, Spill | Medium |
