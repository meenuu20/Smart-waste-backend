import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np
import time
from collections import deque
import struct


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


def remove_event_media(event):
    for key in ("image_path", "video_path"):
        file_path = event.get(key)
        if not file_path:
            continue
        path = Path(file_path)
        if path.exists():
            path.unlink()


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


@app.delete("/api/evidence")
def delete_all_evidence():
    events = load_events()
    for event in events:
        remove_event_media(event)
    save_events([])
    return {"message": "All evidence deleted successfully", "deleted_count": len(events)}


# ===================== STREAMING LOGIC GLOBALS =====================
PERSON_GONE_SECONDS   = 3.0   
STATIONARY_SECONDS    = 2.0   
CARRY_CONFIRM_SECONDS = 0.4   
MOVEMENT_THRESHOLD    = 2    
PERSON_NEAR_DIST      = 450   

VIDEO_DURATION = 20  
PRE_EVENT_SECONDS = 8  
CAMERA_ID = "CAM-01"
CAMERA_LOCATION = "Street1"
VIDEO_EXTENSION = ".mp4"
VIDEO_CODEC = "avc1"
SHOW_DEBUG_WINDOW = os.getenv("SHOW_DEBUG_WINDOW", "").lower() in {"1", "true", "yes", "on"}

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Pi Client Connected via WebSocket")
    
    # Connection-specific state
    camera_fps = 20.0
    pre_event_buffer = deque(maxlen=max(1, int(camera_fps * PRE_EVENT_SECONDS)))
    
    garbage_positions        = {}
    garbage_state            = {}   # 0=unknown 1=carried 2=placed 3=dumped
    garbage_stationary_time  = {}
    garbage_carry_start_time = {}
    garbage_reported         = {}
    garbage_was_carried      = {}
    carry_candidate_logged   = {}
    stationary_logged        = {}
    waiting_for_leave_logged = {}
    last_person_time = time.time()
    image_captured    = {}
    latest_person_garbage_frame = None

    recording = False
    video_writer = None
    record_start_time = None
    current_upload_event = None

    def save_image_local(frame):
        ts = time.strftime("%Y%m%d_%H%M%S")
        image_name = f"dump_{ts}.jpg"
        image_path = IMAGE_DIR / image_name
        cv2.imwrite(str(image_path), frame)
        print(f"[SAVE]  Image: {image_path}")
        return ts, str(image_path), True

    def log_detection_event(level, g_id, message, **context):
        context_str = " ".join(f"{key}={value}" for key, value in context.items())
        suffix = f" {context_str}" if context_str else ""
        print(f"[{level}] garbage_id={g_id} {message}{suffix}")

    def set_garbage_state(g_id, new_state, reason, **context):
        previous_state = garbage_state.get(g_id)
        garbage_state[g_id] = new_state
        previous_label = {0: "unknown", 1: "carried", 2: "placed", 3: "dumped"}.get(previous_state, "unset")
        new_label = {0: "unknown", 1: "carried", 2: "placed", 3: "dumped"}.get(new_state, str(new_state))
        log_detection_event(
            "STATE",
            g_id,
            f"{previous_label} -> {new_label}",
            reason=reason,
            **context,
        )

    def start_recording_local(ts, orig_w, orig_h, current_time, image_path=None, event_timestamp=None, confidence=0.0, details=None):
        nonlocal recording, video_writer, record_start_time, current_upload_event
        video_name = f"dump_{ts}{VIDEO_EXTENSION}"
        video_path = VIDEO_DIR / video_name
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        video_writer = cv2.VideoWriter(str(video_path), fourcc, camera_fps, (orig_w, orig_h))
        
        for buffered_frame in pre_event_buffer:
            video_writer.write(buffered_frame)

        recording = True
        record_start_time = current_time
        current_upload_event = {
            "timestamp": event_timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "camera_id": CAMERA_ID,
            "location": CAMERA_LOCATION,
            "confidence": float(confidence),
            "details": details or {},
            "image_path": str(image_path) if image_path else None,
            "video_path": str(video_path),
        }
        print(f"[SAVE]  Recording with pre-event buffer: {video_path}")

    def finalize_recording_local():
        nonlocal recording, video_writer, record_start_time, current_upload_event
        if not recording: return
        recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        
        # Save to internal DB directly
        if current_upload_event:
            event_id = str(uuid.uuid4())
            event = {
                "id": event_id,
                "timestamp": current_upload_event["timestamp"],
                "camera_id": current_upload_event["camera_id"],
                "location": current_upload_event["location"],
                "confidence": current_upload_event["confidence"],
                "details": current_upload_event["details"],
                "image_path": current_upload_event["image_path"],
                "video_path": current_upload_event["video_path"],
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            events_db = load_events()
            events_db.insert(0, event)
            save_events(events_db)

        record_start_time = None
        current_upload_event = None
        print("[SAVE] Event saved internally to DB")

    try:
        while True:
            # We expect Pi Client to send a single binary message with header + meta + frame
            payload = await websocket.receive_bytes()
            if len(payload) < 8:
                continue
            
            meta_len, frame_len = struct.unpack("!II", payload[:8])
            
            if len(payload) < 8 + meta_len + frame_len:
                print("Incomplete payload received over WS")
                continue

            meta_bytes = payload[8:8+meta_len]
            frame_bytes = payload[8+meta_len:8+meta_len+frame_len]
            
            metadata = json.loads(meta_bytes.decode('utf-8'))
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            orig_w = metadata["orig_w"]
            orig_h = metadata["orig_h"]
            current_time_val = metadata["timestamp"]
            persons = metadata["persons"]
            garbages = metadata["garbages"]
            tracks = metadata["tracks"]

            pre_event_buffer.append(frame.copy())
            annotated = frame.copy()

            for tid_str, track in tracks.items():
                x1, y1, x2, y2 = track["box"]
                cn = track["class_name"]
                color = (0, 255, 0) if cn == "person" else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, f"{cn} {tid_str} {track['conf']:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )

            if persons:
                last_person_time = current_time_val

            if persons and garbages:
                latest_person_garbage_frame = frame.copy()

            person_absent_seconds = current_time_val - last_person_time
            status_text  = "Person: PRESENT" if persons else f"Person absent: {person_absent_seconds:.1f}s"
            status_color = (0, 255, 0) if persons else (0, 165, 255)
            cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            state_labels = {0: "unknown", 1: "carried", 2: "placed", 3: "DUMPED"}
            
            for g_id, g_center in garbages.items():
                garbage_track = tracks[g_id]
                image_path = None

                if g_id not in garbage_state:
                    garbage_state[g_id]    = 0
                    garbage_reported[g_id] = False
                    garbage_was_carried[g_id] = False
                    image_captured[g_id] = False
                    carry_candidate_logged[g_id] = False
                    stationary_logged[g_id] = False
                    waiting_for_leave_logged[g_id] = False
                    log_detection_event("TRACK", g_id, "initialized", state="unknown", confidence=f"{garbage_track['conf']:.2f}")

                movement = np.linalg.norm(np.array(g_center) - np.array(garbage_positions[g_id])) if g_id in garbage_positions else 0
                garbage_positions[g_id] = g_center

                person_near = any(np.linalg.norm(np.array(p) - np.array(g_center)) < PERSON_NEAR_DIST for p in persons.values())
                state = garbage_state[g_id]

                cv2.putText(annotated, f"[{state_labels.get(state,'?')}]", (int(g_center[0]) - 30, int(g_center[1]) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if state == 0:
                    if person_near and movement > MOVEMENT_THRESHOLD:
                        if g_id not in garbage_carry_start_time:
                            garbage_carry_start_time[g_id] = current_time_val
                            if not carry_candidate_logged.get(g_id):
                                log_detection_event(
                                    "CARRY",
                                    g_id,
                                    "carry candidate started",
                                    movement=f"{movement:.2f}",
                                    threshold=MOVEMENT_THRESHOLD,
                                )
                                carry_candidate_logged[g_id] = True
                        if current_time_val - garbage_carry_start_time[g_id] > CARRY_CONFIRM_SECONDS:
                            set_garbage_state(
                                g_id,
                                1,
                                "movement confirmed near person",
                                movement=f"{movement:.2f}",
                                person_near=person_near,
                            )
                            garbage_was_carried[g_id] = True
                            garbage_stationary_time.pop(g_id, None)
                            stationary_logged[g_id] = False
                    elif movement < MOVEMENT_THRESHOLD and garbage_was_carried[g_id]:
                        garbage_carry_start_time.pop(g_id, None)
                        carry_candidate_logged[g_id] = False
                        if g_id not in garbage_stationary_time:
                            garbage_stationary_time[g_id] = current_time_val
                            if not stationary_logged.get(g_id):
                                log_detection_event(
                                    "PLACE",
                                    g_id,
                                    "stationary timer started",
                                    movement=f"{movement:.2f}",
                                    threshold=MOVEMENT_THRESHOLD,
                                )
                                stationary_logged[g_id] = True
                        if current_time_val - garbage_stationary_time[g_id] > STATIONARY_SECONDS:
                            set_garbage_state(
                                g_id,
                                2,
                                "stationary after being carried",
                                stationary_for=f"{current_time_val - garbage_stationary_time[g_id]:.2f}",
                            )
                            waiting_for_leave_logged[g_id] = False
                            if not image_captured[g_id]:
                                snapshot_frame = latest_person_garbage_frame if latest_person_garbage_frame is not None else frame
                                ts, image_path, saved = save_image_local(snapshot_frame)
                                image_captured[g_id] = saved
                                if not recording:
                                    start_recording_local(ts, orig_w, orig_h, current_time_val, image_path, confidence=garbage_track["conf"], details={"garbage_id": g_id, "state": "placed_from_unknown"})
                                elif current_upload_event and not current_upload_event.get("image_path"):
                                    current_upload_event["image_path"] = image_path
                    else:
                        garbage_carry_start_time.pop(g_id, None)
                        garbage_stationary_time.pop(g_id, None)
                        carry_candidate_logged[g_id] = False
                        stationary_logged[g_id] = False

                elif state == 1:
                    if movement < MOVEMENT_THRESHOLD:
                        if g_id not in garbage_stationary_time:
                            garbage_stationary_time[g_id] = current_time_val
                            if not stationary_logged.get(g_id):
                                log_detection_event(
                                    "PLACE",
                                    g_id,
                                    "stationary timer started",
                                    movement=f"{movement:.2f}",
                                    threshold=MOVEMENT_THRESHOLD,
                                )
                                stationary_logged[g_id] = True
                        if current_time_val - garbage_stationary_time[g_id] > STATIONARY_SECONDS:
                            set_garbage_state(
                                g_id,
                                2,
                                "stationary after carried state",
                                stationary_for=f"{current_time_val - garbage_stationary_time[g_id]:.2f}",
                            )
                            waiting_for_leave_logged[g_id] = False
                            if not image_captured[g_id]:
                                snapshot_frame = latest_person_garbage_frame if latest_person_garbage_frame is not None else frame
                                ts, image_path, saved = save_image_local(snapshot_frame)
                                image_captured[g_id] = saved
                                if not recording:
                                    start_recording_local(ts, orig_w, orig_h, current_time_val, image_path, confidence=garbage_track["conf"], details={"garbage_id": g_id, "state": "placed"})
                                elif current_upload_event and not current_upload_event.get("image_path"):
                                    current_upload_event["image_path"] = image_path
                    else:
                        garbage_stationary_time.pop(g_id, None)
                        stationary_logged[g_id] = False

                elif state == 2:
                    if not image_captured[g_id] and latest_person_garbage_frame is not None:
                        _, image_path, saved = save_image_local(latest_person_garbage_frame)
                        image_captured[g_id] = saved
                        if current_upload_event and saved and not current_upload_event.get("image_path"):
                            current_upload_event["image_path"] = image_path

                    remaining = max(0, PERSON_GONE_SECONDS - person_absent_seconds)
                    cv2.putText(
                        annotated, f"Dump in {remaining:.1f}s" if not persons else "Waiting for person to leave",
                        (int(g_center[0]) - 60, int(g_center[1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1
                    )
                    if not persons and not waiting_for_leave_logged.get(g_id):
                        log_detection_event(
                            "WAIT",
                            g_id,
                            "waiting for person absence timer",
                            person_absent=f"{person_absent_seconds:.2f}",
                            required=PERSON_GONE_SECONDS,
                        )
                        waiting_for_leave_logged[g_id] = True
                    elif persons and waiting_for_leave_logged.get(g_id):
                        log_detection_event("WAIT", g_id, "person returned near placed object")
                        waiting_for_leave_logged[g_id] = False

                    if person_absent_seconds >= PERSON_GONE_SECONDS and not garbage_reported[g_id]:
                        set_garbage_state(
                            g_id,
                            3,
                            "person absent threshold reached",
                            person_absent=f"{person_absent_seconds:.2f}",
                        )
                        garbage_reported[g_id] = True
                        if not image_captured[g_id]:
                            snapshot_frame = latest_person_garbage_frame if latest_person_garbage_frame is not None else frame
                            _, image_path, saved = save_image_local(snapshot_frame)
                            image_captured[g_id] = saved
                            if current_upload_event and saved and not current_upload_event.get("image_path"):
                                current_upload_event["image_path"] = image_path
                        if not recording:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            start_recording_local(ts, orig_w, orig_h, current_time_val, image_path if image_captured[g_id] else None, confidence=garbage_track["conf"], details={"garbage_id": g_id, "state": "dumped"})

                elif state == 3:
                    cv2.putText(annotated, "!! DUMPING DETECTED !!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if recording:
                video_writer.write(frame)
                elapsed = time.time() - record_start_time
                cv2.putText(annotated, f"REC {elapsed:.0f}/{VIDEO_DURATION}s", (orig_w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if elapsed > VIDEO_DURATION:
                    finalize_recording_local()

            if SHOW_DEBUG_WINDOW:
                cv2.imshow("Backend WebSocket Stream View", annotated)
                cv2.waitKey(1)

    except WebSocketDisconnect:
        print("Pi Client WebSocket disconnected")
    except Exception as e:
        print(f"Error handling WebSocket stream: {e}")
    finally:
        finalize_recording_local()
        if SHOW_DEBUG_WINDOW:
            cv2.destroyAllWindows()
