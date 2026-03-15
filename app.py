import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
IMAGE_DIR = STORAGE_DIR / "images"
VIDEO_DIR = STORAGE_DIR / "videos"
DB_PATH = BASE_DIR / "evidence_db.json"

for directory in (IMAGE_DIR, VIDEO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not DB_PATH.exists():
    DB_PATH.write_text("[]", encoding="utf-8")

app = FastAPI(title="Dumping Evidence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=STORAGE_DIR), name="media")


def load_events():
    try:
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_events(events):
    DB_PATH.write_text(json.dumps(events, indent=2), encoding="utf-8")


def event_has_existing_media(event):
    image_ok = bool(event.get("image_path") and Path(event["image_path"]).exists())
    video_ok = bool(event.get("video_path") and Path(event["video_path"]).exists())
    return image_ok or video_ok


def parse_details(raw_details):
    if not raw_details:
        return {}
    try:
        return json.loads(raw_details)
    except json.JSONDecodeError:
        return {"summary": raw_details}


def save_upload_file(upload: UploadFile, target_dir: Path, event_id: str):
    suffix = Path(upload.filename or "").suffix or ""
    safe_name = f"{event_id}_{upload.filename}" if upload.filename else f"{event_id}{suffix}"
    destination = target_dir / safe_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return destination


def serialize_event(event):
    image_path = event.get("image_path")
    video_path = event.get("video_path")
    image_name = Path(image_path).name if image_path else None
    video_name = Path(video_path).name if video_path else None
    return {
        **event,
        "image_url": f"/media/images/{image_name}" if image_name else None,
        "video_url": f"/media/videos/{video_name}" if video_name else None,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/evidence", status_code=201)
async def create_evidence(
    timestamp: str = Form(...),
    camera_id: str = Form(...),
    location: str = Form(...),
    confidence: float = Form(...),
    details: str = Form(""),
    image: UploadFile | None = File(default=None),
    video: UploadFile | None = File(default=None),
):
    if image is None and video is None:
        raise HTTPException(status_code=400, detail="At least one evidence file is required")

    event_id = str(uuid.uuid4())
    image_path = str(save_upload_file(image, IMAGE_DIR, event_id)) if image is not None else None
    video_path = str(save_upload_file(video, VIDEO_DIR, event_id)) if video is not None else None

    event = {
        "id": event_id,
        "timestamp": timestamp,
        "camera_id": camera_id,
        "location": location,
        "confidence": confidence,
        "details": parse_details(details),
        "image_path": image_path,
        "video_path": video_path,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    events = load_events()
    events.insert(0, event)
    save_events(events)
    return {"message": "Evidence stored successfully", "event": serialize_event(event)}


@app.get("/api/evidence")
def list_evidence():
    valid_events = [event for event in load_events() if event_has_existing_media(event)]
    if len(valid_events) != len(load_events()):
        save_events(valid_events)
    events = [serialize_event(event) for event in valid_events]
    return {"items": events, "count": len(events)}


@app.get("/api/evidence/{event_id}")
def get_evidence(event_id: str):
    for event in load_events():
        if event["id"] == event_id:
            if not event_has_existing_media(event):
                raise HTTPException(status_code=404, detail="Evidence files are missing")
            return serialize_event(event)
    raise HTTPException(status_code=404, detail="Evidence not found")
