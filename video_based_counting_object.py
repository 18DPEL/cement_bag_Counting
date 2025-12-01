import cv2
import json
import os
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# Configuration
# =========================
VIDEO_SOURCE = r"rtsp://localhost:8554/mystream"  # change to your file path or RTSP url
MODEL_PATH   = r"C:\Users\ayuba\Downloads\best (15).pt"  # your YOLOv11n model (custom .pt supported)
CLASS_FILTER = None  # e.g. [0] if your bag class id is 0; else None = all classes
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
TRACKER      = "bytetrack.yaml"
CFG_FILE     = "roi_config.json"

# Direction options:
# axis = 'x' means we consider motion left<->right across the ROI (use for belts running roughly left-right)
# axis = 'y' means we consider motion up<->down across the ROI (use for belts running roughly up-down)
axis = 'y'  # default; toggle with 'a'
forward_positive = True  # if True, A->B = +1; else A->B = -1 (flip with 'f')

# =========================
# Globals for ROI editing
# =========================
roi = None  # [x1,y1,x2,y2] with x1<x2, y1<y2
edit_mode = True
dragging = False
resizing = False
drag_offset = (0,0)
resize_corner = None
handle_size = 10

paused = False

# =========================
# Per-object state for crossing logic
# =========================
# We track simple state machine for each ID:
# state: 'outside' or 'inside'
# entry_side: 'A' or 'B' (which side it entered from along chosen axis)
obj_state = {}  # id -> dict(state='outside', entry_side=None)
count_value = 0

# =========================
# Helpers
# =========================
def normalize_roi(r):
    if r is None:
        return None
    x1,y1,x2,y2 = r
    if x1>x2: x1,x2 = x2,x1
    if y1>y2: y1,y2 = y2,y1
    return [x1,y1,x2,y2]

def point_in_rect(pt, r):
    if r is None: return False
    x,y = pt
    x1,y1,x2,y2 = r
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def corner_hit(pt, r, size=10):
    """Return which corner is hit or None. Corners: 'tl','tr','bl','br'."""
    if r is None: return None
    x,y = pt
    x1,y1,x2,y2 = r
    corners = {
        'tl': (x1,y1),
        'tr': (x2,y1),
        'bl': (x1,y2),
        'br': (x2,y2),
    }
    for name,(cx,cy) in corners.items():
        if abs(x-cx)<=size and abs(y-cy)<=size:
            return name
    return None

def move_roi(r, dx, dy, w, h):
    x1,y1,x2,y2 = r
    nx1, ny1 = max(0, min(w-1, x1+dx)), max(0, min(h-1, y1+dy))
    nx2, ny2 = max(0, min(w-1, x2+dx)), max(0, min(h-1, y2+dy))
    return [nx1, ny1, nx2, ny2]

def resize_roi(r, corner, pt, frame_w, frame_h, min_size=20):
    x1,y1,x2,y2 = r
    x,y = pt
    if corner=='tl':
        x1,y1 = x,y
    elif corner=='tr':
        x2,y1 = x,y
    elif corner=='bl':
        x1,y2 = x,y
    elif corner=='br':
        x2,y2 = x,y
    # clamp
    x1 = max(0, min(frame_w-1, x1))
    x2 = max(0, min(frame_w-1, x2))
    y1 = max(0, min(frame_h-1, y1))
    y2 = max(0, min(frame_h-1, y2))
    r2 = normalize_roi([x1,y1,x2,y2])
    # enforce min size
    if r2 and (r2[2]-r2[0] < min_size or r2[3]-r2[1] < min_size):
        return r  # ignore tiny
    return r2

def draw_roi(frame, r, color=(0,255,0)):
    if r is None: return
    x1,y1,x2,y2 = r
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    # draw handles
    for (cx,cy) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        cv2.rectangle(frame, (cx-handle_size, cy-handle_size), (cx+handle_size, cy+handle_size), color, -1)
    # mark A/B sides according to axis
    if axis=='x':
        # Left = A, Right = B
        cv2.putText(frame, "A", (x1+5, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, "B", (x2-20, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    else:
        # Top = A, Bottom = B
        cv2.putText(frame, "A", ((x1+x2)//2-10, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, "B", ((x1+x2)//2-10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

def save_cfg():
    data = {
        "roi": roi,
        "axis": axis,
        "forward_positive": forward_positive
    }
    with open(CFG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {CFG_FILE}")

def load_cfg():
    global roi, axis, forward_positive
    if os.path.exists(CFG_FILE):
        with open(CFG_FILE, "r") as f:
            data = json.load(f)
        roi = normalize_roi(data.get("roi"))
        axis = data.get("axis", axis)
        forward_positive = data.get("forward_positive", forward_positive)
        print(f"[INFO] Loaded {CFG_FILE}")

# =========================
# Mouse callback
# =========================
mouse_down_pt = None
def on_mouse(event, x, y, flags, param):
    global roi, edit_mode, dragging, drag_offset, resizing, resize_corner, mouse_down_pt

    if not edit_mode:
        return

    frame_w, frame_h = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if roi is not None:
            hit = corner_hit((x,y), roi, handle_size)
            if hit:
                resizing = True
                resize_corner = hit
                return
            if point_in_rect((x,y), roi):
                dragging = True
                drag_offset = (x - roi[0], y - roi[1])
                return
        # start drawing a new ROI
        mouse_down_pt = (x,y)
        roi = [x,y,x,y]

    elif event == cv2.EVENT_MOUSEMOVE:
        if resizing and roi is not None:
            roi[:] = resize_roi(roi, resize_corner, (x,y), frame_w, frame_h)
        elif dragging and roi is not None:
            dx = x - (roi[0] + drag_offset[0])
            dy = y - (roi[1] + drag_offset[1])
            roi[:] = move_roi(roi, dx, dy, frame_w, frame_h)
        elif mouse_down_pt is not None:
            x0,y0 = mouse_down_pt
            roi = normalize_roi([x0,y0,x,y])

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        resizing = False
        resize_corner = None
        mouse_down_pt = None

# =========================
# Crossing logic
# =========================
def side_of_rect(pt, r, axis):
    """Return 'inside', 'A', or 'B' relative to ROI along chosen axis."""
    if r is None:
        return 'outside'
    x,y = pt
    x1,y1,x2,y2 = r
    inside = (x1 <= x <= x2) and (y1 <= y <= y2)
    if inside:
        return 'inside'
    if axis=='x':
        return 'A' if x < x1 else ('B' if x > x2 else 'inside')
    else:
        return 'A' if y < y1 else ('B' if y > y2 else 'inside')

def update_counter(track_id, pt):
    """Finite-state machine per track to count A->B (forward) or B->A (backward)."""
    global count_value
    st = obj_state.setdefault(track_id, {'state':'outside', 'entry_side':None})
    pos = side_of_rect(pt, roi, axis)

    if st['state'] == 'outside':
        if pos == 'inside':
            # entering, decide which side it came from based on where its center is now
            # If just at the edge we infer from proximity
            # Better: look at last side (A/B); for simplicity we infer by nearest side
            # We'll store whichever side it's closer to
            if axis=='x':
                x = pt[0]; x1,y1,x2,y2 = roi
                st['entry_side'] = 'A' if abs(x - x1) < abs(x - x2) else 'B'
            else:
                y = pt[1]; x1,y1,x2,y2 = roi
                st['entry_side'] = 'A' if abs(y - y1) < abs(y - y2) else 'B'
            st['state'] = 'inside'

    elif st['state'] == 'inside':
        if pos in ('A','B'):
            exit_side = pos
            if st['entry_side'] and exit_side != st['entry_side']:
                # traversed fully across ROI → count
                if forward_positive:
                    delta = +1 if (st['entry_side']=='A' and exit_side=='B') else -1
                else:
                    delta = -1 if (st['entry_side']=='A' and exit_side=='B') else +1
                count_value += delta
            # reset
            st['state'] = 'outside'
            st['entry_side'] = None

# =========================
# Main
# =========================
def main():
    # ── globals used inside main (declare once here, not inside key blocks)
    global roi, obj_state, axis, forward_positive, paused, count_value, edit_mode

    load_cfg()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Unable to open video source:", VIDEO_SOURCE)
        return

    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Could not read first frame.")
        return
    H, W = frame.shape[:2]

    cv2.namedWindow("Belt Counter", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Belt Counter", on_mouse, (W, H))

    # Load YOLO
    model = YOLO(MODEL_PATH)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect + Track
            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False
            )

            if len(results) > 0:
                r = results[0]
                if r.boxes is not None and r.boxes.id is not None:
                    ids  = r.boxes.id.cpu().tolist()
                    xyxy = r.boxes.xyxy.cpu().tolist()
                    cls  = r.boxes.cls.cpu().tolist() if r.boxes.cls is not None else [None]*len(ids)

                    for (x1,y1,x2,y2), tid, c in zip(xyxy, ids, cls):
                        if CLASS_FILTER is not None and (c is None or int(c) not in CLASS_FILTER):
                            continue

                        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Update the counter FSM when ROI exists
                        if roi is not None:
                            update_counter(int(tid), (cx, cy))

                        # Draw track box & center
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID {int(tid)}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # ── UI overlay
        if roi is not None:
            draw_roi(frame, roi, (0, 255, 0))

        cv2.putText(frame, f"Count: {count_value}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.putText(frame,
                    f"Edit:{'ON' if edit_mode else 'OFF'}  Axis:{axis}  Forward:+1 is {'A->B' if forward_positive else 'B->A'}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame,
                    "Keys: e-edit  a-axis  f-flip  s-save  l-load  c-clear  p-pause  q-quit",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Belt Counter", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── hotkeys
        if key == ord('q'):
            break
        elif key == ord('e'):
            edit_mode = not edit_mode
        elif key == ord('a'):
            axis = 'x' if axis == 'y' else 'y'
        elif key == ord('f'):
            forward_positive = not forward_positive
        elif key == ord('c'):
            # Clear ROI and per-object states (no extra global here!)
            roi = None
            obj_state.clear()
            count_value = 0
        elif key == ord('s'):
            save_cfg()
        elif key == ord('l'):
            load_cfg()
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
