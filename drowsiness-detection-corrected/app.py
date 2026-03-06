import argparse
import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pygame import mixer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--model",
        type=str,
        default="dl_model/drowsiness_cnn.h5",
        help="Path to eye-state CNN model (.h5). If missing, conservative fallback is used.",
    )
    parser.add_argument(
        "--alert-seconds",
        type=float,
        default=2.0,
        help="Continuous closed-eye time in seconds needed to trigger alert",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.9,
        help="How quickly closed time decays when eyes are open (seconds/second)",
    )
    parser.add_argument(
        "--fallback-confirm-frames",
        type=int,
        default=3,
        help="Consecutive fallback evidence frames required before counting as closed",
    )
    parser.add_argument(
        "--fallback-eye-ar-threshold",
        type=float,
        default=0.30,
        help="Absolute fallback closed-eye threshold for eye box aspect ratio (height/width)",
    )
    parser.add_argument(
        "--fallback-ref-ratio",
        type=float,
        default=0.78,
        help="Adaptive closed threshold ratio against running open-eye reference",
    )
    parser.add_argument(
        "--close-prob-threshold",
        type=float,
        default=0.60,
        help="CNN closed-eye probability threshold to switch into closed state",
    )
    parser.add_argument(
        "--open-prob-threshold",
        type=float,
        default=0.40,
        help="CNN closed-eye probability threshold to switch back into open state",
    )
    parser.add_argument(
        "--eye-prob-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing factor for eye closed probability (0..1)",
    )
    parser.add_argument(
        "--cnn-confirm-frames",
        type=int,
        default=4,
        help="Consecutive CNN-closed frames required before counting as closed.",
    )
    parser.add_argument(
        "--open-reset-frames",
        type=int,
        default=3,
        help="Consecutive CNN-open frames required to hard-reset closed timer.",
    )
    parser.add_argument(
        "--startup-warmup-seconds",
        type=float,
        default=2.0,
        help="Disable closed counting/alarm for first N seconds after start.",
    )
    parser.add_argument(
        "--startup-valid-eye-frames",
        type=int,
        default=12,
        help="Require this many valid two-eye CNN frames before enabling closed counting.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Auto-calibrate probability thresholds from startup open-eye frames.",
    )
    parser.add_argument(
        "--model-closed-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which model output index represents Closed class (0 or 1).",
    )
    parser.add_argument(
        "--save-alert-frames",
        action="store_true",
        help="Save a frame snapshot each time an alert is fired.",
    )
    parser.add_argument(
        "--alert-dir",
        type=str,
        default="alerts",
        help="Folder for saved alert frames (relative to app.py unless absolute).",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="Optional CSV file path for runtime event logs.",
    )
    parser.add_argument(
        "--voice-status",
        action="store_true",
        help="Enable voice announcements for safe-driving and sleeping states (requires pyttsx3).",
    )
    parser.add_argument(
        "--voice-safe",
        action="store_true",
        help="Enable safe-driving voice message. Disabled by default.",
    )
    parser.add_argument(
        "--voice-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between repeated voice announcements.",
    )
    parser.add_argument(
        "--sleep-voice-repeats",
        type=int,
        default=3,
        help="How many times to repeat sleep warning per alert trigger.",
    )
    parser.add_argument(
        "--sound-status",
        action="store_true",
        help="Enable WAV sound announcements.",
    )
    parser.add_argument(
        "--sleep-sound",
        type=str,
        default="sleep_alert.wav",
        help="WAV file for sleeping alert.",
    )
    parser.add_argument(
        "--safe-sound",
        type=str,
        default="safe_status.wav",
        help="WAV file for safe-driving status.",
    )
    parser.add_argument(
        "--sound-interval",
        type=float,
        default=5.0,
        help="Minimum seconds between repeated sound announcements.",
    )
    parser.add_argument(
        "--safe-confirm-frames",
        type=int,
        default=8,
        help="Consecutive non-drowsy frames required before safe sound can play.",
    )
    parser.add_argument("--cooldown", type=float, default=1.0, help="Alarm cooldown in seconds")
    return parser.parse_args()


def load_cascades(base: Path):
    face_primary = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_alt.xml"))
    face_backup = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    leye = cv2.CascadeClassifier(str(base / "haarcascade_lefteye_2splits.xml"))
    reye = cv2.CascadeClassifier(str(base / "haarcascade_righteye_2splits.xml"))
    generic_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    if face_primary.empty() or face_backup.empty() or leye.empty() or reye.empty() or generic_eye.empty():
        raise RuntimeError("Failed to load one or more Haar cascades.")
    return face_primary, face_backup, leye, reye, generic_eye


def preprocess_eye(eye_bgr: np.ndarray) -> np.ndarray:
    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    eye_rgb = cv2.resize(eye_rgb, (32, 32), interpolation=cv2.INTER_AREA)
    eye_arr = eye_rgb.astype(np.float32) / 255.0
    return eye_arr.reshape((1, 32, 32, 3))


def pick_largest(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def eye_aspect_ratio(box) -> float:
    if box is None:
        return 0.0
    w = max(1, int(box[2]))
    h = max(1, int(box[3]))
    return float(h) / float(w)


def detect_face(gray, face_primary, face_backup):
    faces = face_primary.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        faces = face_backup.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
    return faces


def update_ref(curr: float | None, value: float, alpha: float = 0.06) -> float:
    if curr is None:
        return value
    return (1.0 - alpha) * curr + alpha * value


def infer_closed_from_box(box, ar_ref, args) -> bool:
    if box is None:
        return False
    ar = eye_aspect_ratio(box)
    adaptive_th = (ar_ref or ar) * args.fallback_ref_ratio
    return ar < min(args.fallback_eye_ar_threshold, adaptive_th)


def update_ema(curr: float | None, value: float, alpha: float) -> float:
    if curr is None:
        return value
    return (1.0 - alpha) * curr + alpha * value


def decide_hysteresis(curr_prob: float, prev_state: bool | None, close_th: float, open_th: float) -> bool:
    if curr_prob >= close_th:
        return True
    if curr_prob <= open_th:
        return False
    if prev_state is not None:
        return prev_state
    return curr_prob >= 0.5


def draw_eye_state_overlay(frame, origin_x, origin_y, box, is_closed):
    if box is None:
        return
    x, y, w, h = box
    cx = origin_x + x + (w // 2)
    cy = origin_y + y + (h // 2)
    ax = max(4, int(w * 0.45))
    ay = max(3, int(h * 0.35))

    if is_closed is None:
        fill_color = (0, 255, 255)  # yellow while unknown/tracking
    elif is_closed:
        fill_color = (0, 0, 255)  # red: closed
    else:
        fill_color = (0, 255, 0)  # green: open

    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, fill_color, thickness=-1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (255, 255, 255), thickness=1)


class VoiceAnnouncer:
    def __init__(self, enabled: bool, interval: float):
        self.enabled = enabled
        self.interval = max(0.5, interval)
        self.last_said = {"safe": 0.0, "sleep": 0.0}
        self._q: queue.Queue[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = False
        if not enabled:
            return
        try:
            import pyttsx3

            engine = pyttsx3.init()
            self._q = queue.Queue()

            def run():
                while not self._stop:
                    try:
                        msg, repeats, gap_sec = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    for i in range(max(1, repeats)):
                        engine.say(msg)
                        engine.runAndWait()
                        if i < repeats - 1 and gap_sec > 0:
                            time.sleep(gap_sec)

            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
        except Exception:
            self.enabled = False
            print("Voice disabled: install pyttsx3 to use --voice-status")

    def say(
        self,
        key: str,
        text: str,
        priority: bool = False,
        force: bool = False,
        repeats: int = 1,
        gap_sec: float = 0.0,
    ):
        if not self.enabled or self._q is None:
            return
        now = time.time()
        if (not force) and (now - self.last_said.get(key, 0.0) < self.interval):
            return
        self.last_said[key] = now
        if priority:
            try:
                while True:
                    self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put((text, max(1, int(repeats)), max(0.0, float(gap_sec))))

    def close(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent

    face_primary, face_backup, left_eye_cascade, right_eye_cascade, generic_eye_cascade = load_cascades(base)

    model = None
    model_path = base / args.model
    if model_path.exists():
        from tensorflow.keras.models import load_model

        model = load_model(str(model_path))
        print(f"Loaded model: {model_path}")
    else:
        print(
            "Model file not found. Running fallback mode without CNN. "
            "(Using permissive eye detection + repeated evidence.)"
        )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    closed_time_seconds = 0.0
    border_thickness = 2
    last_alarm_time = 0.0
    alert_frame_count = 0
    fallback_streak = 0
    left_ar_ref = None
    right_ar_ref = None
    left_closed_prob_ema = None
    right_closed_prob_ema = None
    left_closed_state = None
    right_closed_state = None
    cnn_closed_streak = 0
    cnn_open_streak = 0
    startup_valid_frames = 0
    startup_open_probs = []
    calibrated_once = False
    last_any_eye_time = 0.0
    prev_t = time.perf_counter()
    app_start_t = time.time()
    fps_ema = 0.0
    prev_status = None
    prev_alert_active = False

    alert_dir = Path(args.alert_dir)
    if not alert_dir.is_absolute():
        alert_dir = base / alert_dir
    if args.save_alert_frames:
        alert_dir.mkdir(parents=True, exist_ok=True)

    log_writer = None
    log_file = None
    if args.log_csv:
        log_path = Path(args.log_csv)
        if not log_path.is_absolute():
            log_path = base / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if log_file.tell() == 0:
            log_writer.writerow(
                [
                    "timestamp",
                    "event",
                    "state",
                    "closed_time",
                    "left_closed_prob",
                    "right_closed_prob",
                    "cnn_closed_streak",
                    "cnn_open_streak",
                    "fallback_streak",
                    "face_detected",
                ]
            )
    announcer = VoiceAnnouncer(enabled=args.voice_status, interval=args.voice_interval)
    sleep_sound = None
    safe_sound = None
    last_sleep_sound_time = 0.0
    last_safe_sound_time = 0.0
    safe_announced_once = False
    sleep_loop_active = False
    if args.sound_status:
        try:
            mixer.init()
            sleep_sound_path = Path(args.sleep_sound)
            if not sleep_sound_path.is_absolute():
                sleep_sound_path = base / sleep_sound_path
            safe_sound_path = Path(args.safe_sound)
            if not safe_sound_path.is_absolute():
                safe_sound_path = base / safe_sound_path

            if sleep_sound_path.exists():
                sleep_sound = mixer.Sound(str(sleep_sound_path))
            else:
                print(f"Sleep sound not found: {sleep_sound_path}")
            if safe_sound_path.exists():
                safe_sound = mixer.Sound(str(safe_sound_path))
            else:
                print(f"Safe sound not found: {safe_sound_path}")
        except Exception as e:
            print(f"Sound disabled: {e}")

    while True:
        ok, frame = cap.read()
        now_t = time.perf_counter()
        dt = now_t - prev_t
        prev_t = now_t
        if dt < 0.0:
            dt = 0.0
        if dt > 0.25:
            dt = 0.25
        if dt > 0:
            fps_now = 1.0 / dt
            fps_ema = fps_now if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * fps_now)

        if not ok:
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = detect_face(gray, face_primary, face_backup)

        both_closed = False
        both_green = False
        status = "No Face"
        left_detected = 0
        right_detected = 0
        left_ar = 0.0
        right_ar = 0.0
        left_closed = None
        right_closed = None
        force_open_reset = False

        full_eyes = generic_eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=2, minSize=(14, 14)
        )
        best_face = pick_largest(faces)
        if best_face is not None:
            fx, fy, fw, fh = best_face
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 220, 255), 2)

            upper_y_end = fy + int(0.65 * fh)
            face_roi_gray = gray[fy:upper_y_end, fx : fx + fw]
            face_roi_bgr = frame[fy:upper_y_end, fx : fx + fw]

            left_eyes = left_eye_cascade.detectMultiScale(
                face_roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(14, 14)
            )
            right_eyes = right_eye_cascade.detectMultiScale(
                face_roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(14, 14)
            )

            left_eye_box = pick_largest(left_eyes)
            right_eye_box = pick_largest(right_eyes)

            generic_eyes = generic_eye_cascade.detectMultiScale(
                face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(14, 14)
            )
            generic_sorted = sorted(generic_eyes, key=lambda b: b[2] * b[3], reverse=True)
            if len(generic_sorted) >= 2 and (left_eye_box is None or right_eye_box is None):
                two = sorted(generic_sorted[:2], key=lambda b: b[0])
                if left_eye_box is None:
                    left_eye_box = two[0]
                if right_eye_box is None:
                    right_eye_box = two[1]

            if left_eye_box is not None:
                lx, ly, lw, lh = left_eye_box
                left_detected = 1
                left_ar = eye_aspect_ratio(left_eye_box)
                if left_ar > 0.12:
                    left_ar_ref = update_ref(left_ar_ref, left_ar)
            if right_eye_box is not None:
                rx, ry, rw, rh = right_eye_box
                right_detected = 1
                right_ar = eye_aspect_ratio(right_eye_box)
                if right_ar > 0.12:
                    right_ar_ref = update_ref(right_ar_ref, right_ar)

            if model is not None:
                fallback_streak = 0
                if left_eye_box is not None and right_eye_box is not None:
                    lx, ly, lw, lh = left_eye_box
                    rx, ry, rw, rh = right_eye_box
                    l_eye = face_roi_bgr[ly : ly + lh, lx : lx + lw]
                    r_eye = face_roi_bgr[ry : ry + rh, rx : rx + rw]
                    if l_eye.size > 0 and r_eye.size > 0:
                        startup_valid_frames += 1
                        l_probs = model.predict(preprocess_eye(l_eye), verbose=0)[0]
                        r_probs = model.predict(preprocess_eye(r_eye), verbose=0)[0]
                        l_closed_prob = float(l_probs[args.model_closed_index])
                        r_closed_prob = float(r_probs[args.model_closed_index])
                        if not calibrated_once and (time.time() - app_start_t) <= max(0.0, args.startup_warmup_seconds):
                            startup_open_probs.append(l_closed_prob)
                            startup_open_probs.append(r_closed_prob)

                        left_closed_prob_ema = update_ema(left_closed_prob_ema, l_closed_prob, args.eye_prob_alpha)
                        right_closed_prob_ema = update_ema(right_closed_prob_ema, r_closed_prob, args.eye_prob_alpha)

                        left_closed_state = decide_hysteresis(
                            left_closed_prob_ema,
                            left_closed_state,
                            args.close_prob_threshold,
                            args.open_prob_threshold,
                        )
                        right_closed_state = decide_hysteresis(
                            right_closed_prob_ema,
                            right_closed_state,
                            args.close_prob_threshold,
                            args.open_prob_threshold,
                        )

                        left_closed = left_closed_state
                        right_closed = right_closed_state
                        if left_closed and right_closed:
                            cnn_closed_streak += 1
                            cnn_open_streak = 0
                        else:
                            cnn_closed_streak = 0
                            cnn_open_streak += 1
                        both_closed = cnn_closed_streak >= max(1, args.cnn_confirm_frames)
                        if cnn_open_streak >= max(1, args.open_reset_frames):
                            force_open_reset = True
                        status = "Closed" if both_closed else "Open"
                    else:
                        cnn_closed_streak = 0
                        cnn_open_streak = 0
                        left_closed_state = None
                        right_closed_state = None
                        status = "Tracking"
                else:
                    cnn_closed_streak = 0
                    cnn_open_streak = 0
                    left_closed_state = None
                    right_closed_state = None
                    status = "Tracking"
            else:
                left_closed = infer_closed_from_box(left_eye_box, left_ar_ref, args) if left_detected else False
                right_closed = infer_closed_from_box(right_eye_box, right_ar_ref, args) if right_detected else False

                closed_evidence = False
                if left_detected and right_detected:
                    closed_evidence = left_closed and right_closed
                elif left_detected and not right_detected:
                    closed_evidence = left_closed
                elif right_detected and not left_detected:
                    closed_evidence = right_closed
                elif not left_detected and not right_detected:
                    closed_evidence = True

                if left_detected or right_detected:
                    last_any_eye_time = time.time()

                if closed_evidence:
                    fallback_streak += 1
                else:
                    fallback_streak = max(0, fallback_streak - 1)

                both_closed = fallback_streak >= max(1, args.fallback_confirm_frames)
                if both_closed:
                    status = "Closed*"
                elif closed_evidence:
                    status = "Verifying*"
                else:
                    status = "Open*"

            # Eye-shape overlays: green=open, red=closed.
            draw_eye_state_overlay(frame, fx, fy, left_eye_box, left_closed if left_detected else None)
            draw_eye_state_overlay(frame, fx, fy, right_eye_box, right_closed if right_detected else None)
        else:
            if model is None:
                # Face detector can drop intermittently on low-end webcams.
                # Use full-frame eye detection and recent-eye memory.
                full_eye_boxes = sorted(full_eyes, key=lambda b: b[2] * b[3], reverse=True)[:2]
                for ex, ey, ew, eh in full_eye_boxes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (180, 180, 0), 1)

                closed_evidence = False
                if len(full_eye_boxes) >= 2:
                    two = sorted(full_eye_boxes, key=lambda b: b[0])
                    l_box = two[0]
                    r_box = two[1]
                    l_ar = eye_aspect_ratio(l_box)
                    r_ar = eye_aspect_ratio(r_box)
                    if l_ar > 0.12:
                        left_ar_ref = update_ref(left_ar_ref, l_ar)
                    if r_ar > 0.12:
                        right_ar_ref = update_ref(right_ar_ref, r_ar)
                    left_closed = infer_closed_from_box(l_box, left_ar_ref, args)
                    right_closed = infer_closed_from_box(r_box, right_ar_ref, args)
                    closed_evidence = left_closed and right_closed
                    last_any_eye_time = time.time()
                    status = "Open*"
                else:
                    # If eyes were tracked recently and disappear, treat as closure evidence.
                    closed_evidence = (time.time() - last_any_eye_time) < 1.8
                    status = "Verifying*"

                if closed_evidence:
                    fallback_streak += 1
                else:
                    fallback_streak = max(0, fallback_streak - 1)

                both_closed = fallback_streak >= max(1, args.fallback_confirm_frames)
                if both_closed:
                    status = "Closed*"
            else:
                fallback_streak = 0
                left_closed_state = None
                right_closed_state = None
                cnn_closed_streak = 0
                cnn_open_streak = 0

        both_green = (
            left_detected == 1
            and right_detected == 1
            and left_closed is False
            and right_closed is False
        )

        if force_open_reset:
            closed_time_seconds = 0.0
        else:
            warmup_elapsed = (time.time() - app_start_t) >= max(0.0, args.startup_warmup_seconds)
            warmup_frames_ok = startup_valid_frames >= max(0, args.startup_valid_eye_frames)
            ready_for_counting = warmup_elapsed and warmup_frames_ok
            if both_closed and ready_for_counting:
                closed_time_seconds += dt
            elif not both_closed:
                closed_time_seconds = max(0.0, closed_time_seconds - (args.decay_rate * dt))
            else:
                # During startup warmup, never accumulate closed-time.
                closed_time_seconds = 0.0

        cv2.rectangle(frame, (0, height - 85), (760, height), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(frame, f"State: {status}", (10, height - 60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Closed Time: {closed_time_seconds:.1f}s / {args.alert_seconds:.1f}s",
            (10, height - 40),
            font,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS:{fps_ema:.1f} Face:{1 if best_face is not None else 0} L:{left_detected} R:{right_detected} LAR:{left_ar:.2f} RAR:{right_ar:.2f} LRef:{(left_ar_ref or 0):.2f} RRef:{(right_ar_ref or 0):.2f} Lp:{(left_closed_prob_ema or 0):.2f} Rp:{(right_closed_prob_ema or 0):.2f} CStreak:{cnn_closed_streak} OStreak:{cnn_open_streak} FStreak:{fallback_streak}",
            (10, height - 20),
            font,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        if model is None:
            cv2.putText(frame, "Fallback mode (no CNN model)", (10, 25), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            warmup_left = max(0.0, args.startup_warmup_seconds - (time.time() - app_start_t))
            if args.auto_calibrate and (not calibrated_once) and warmup_left <= 0.0 and len(startup_open_probs) >= 20:
                p95 = float(np.percentile(np.array(startup_open_probs, dtype=np.float32), 95))
                new_close = max(args.close_prob_threshold, min(0.90, p95 + 0.20))
                new_open = min(new_close - 0.10, max(0.10, p95 + 0.05))
                args.close_prob_threshold = new_close
                args.open_prob_threshold = max(0.05, min(new_open, args.open_prob_threshold))
                calibrated_once = True
                print(
                    f"Auto-calibrated thresholds: close={args.close_prob_threshold:.2f}, "
                    f"open={args.open_prob_threshold:.2f}, p95_open_closedprob={p95:.2f}"
                )
            if warmup_left > 0.0 or startup_valid_frames < max(0, args.startup_valid_eye_frames):
                cv2.putText(
                    frame,
                    f"Warmup: {warmup_left:.1f}s  ValidEyeFrames: {startup_valid_frames}/{args.startup_valid_eye_frames}",
                    (10, 25),
                    font,
                    0.9,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if closed_time_seconds >= args.alert_seconds:
            now = time.time()
            if now - last_alarm_time >= args.cooldown:
                last_alarm_time = now
                announcer.say(
                    "sleep",
                    "Driver is sleeping. Please wake up.",
                    priority=True,
                    force=True,
                    repeats=max(1, args.sleep_voice_repeats),
                    gap_sec=0.35,
                )
                if args.save_alert_frames:
                    alert_file = alert_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alert_frame_count:04d}.jpg"
                    cv2.imwrite(str(alert_file), frame)
                    alert_frame_count += 1
                if log_writer is not None:
                    log_writer.writerow(
                        [
                            datetime.now().isoformat(timespec="seconds"),
                            "alert",
                            status,
                            f"{closed_time_seconds:.3f}",
                            f"{(left_closed_prob_ema or 0):.4f}",
                            f"{(right_closed_prob_ema or 0):.4f}",
                            cnn_closed_streak,
                            cnn_open_streak,
                            fallback_streak,
                            1 if best_face is not None else 0,
                        ]
                    )

            border_thickness = border_thickness + 2 if border_thickness < 16 else 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), border_thickness)
            cv2.putText(frame, "DROWSINESS ALERT", (10, 55), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            alert_active = True
            sleep_loop_active = True
        else:
            alert_active = False

        if log_writer is not None and status != prev_status:
            log_writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    "state_change",
                    status,
                    f"{closed_time_seconds:.3f}",
                    f"{(left_closed_prob_ema or 0):.4f}",
                    f"{(right_closed_prob_ema or 0):.4f}",
                    cnn_closed_streak,
                    cnn_open_streak,
                    fallback_streak,
                    1 if best_face is not None else 0,
                ]
            )
        if log_writer is not None and alert_active != prev_alert_active:
            log_writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    "alert_on" if alert_active else "alert_off",
                    status,
                    f"{closed_time_seconds:.3f}",
                    f"{(left_closed_prob_ema or 0):.4f}",
                    f"{(right_closed_prob_ema or 0):.4f}",
                    cnn_closed_streak,
                    cnn_open_streak,
                    fallback_streak,
                    1 if best_face is not None else 0,
                ]
            )
        prev_status = status
        prev_alert_active = alert_active

        if args.voice_safe and (not alert_active) and status.startswith("Open"):
            announcer.say("safe", "Driver is safely driving.")

        # Play safe sound only once when initial stable open-eye detection is available.
        if safe_sound is not None and (not safe_announced_once) and both_green:
            now = time.time()
            if (now - last_safe_sound_time) >= 0.5:
                safe_sound.play()
                last_safe_sound_time = now
                safe_announced_once = True

        # After first drowsiness trigger, repeat sleep sound until both eyes turn green.
        if sleep_loop_active and sleep_sound is not None:
            now = time.time()
            if (now - last_sleep_sound_time) >= args.sound_interval:
                sleep_sound.play()
                last_sleep_sound_time = now
        if sleep_loop_active and both_green:
            sleep_loop_active = False

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if log_file is not None:
        log_file.close()
    announcer.close()


if __name__ == "__main__":
    main()
