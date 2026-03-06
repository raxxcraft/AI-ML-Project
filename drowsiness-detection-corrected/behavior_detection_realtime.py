import argparse
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class VoiceAnnouncer:
    def __init__(self, enabled: bool, interval: float):
        self.enabled = enabled
        self.interval = max(0.5, interval)
        self.last_said: dict[str, float] = {}
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
                        msg = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    engine.say(msg)
                    engine.runAndWait()

            self._thread = threading.Thread(target=run, daemon=True)
            self._thread.start()
        except Exception:
            self.enabled = False
            print("Voice disabled: install pyttsx3")

    def say(self, key: str, text: str):
        if not self.enabled or self._q is None:
            return
        now = time.time()
        if now - self.last_said.get(key, 0.0) < self.interval:
            return
        self.last_said[key] = now
        self._q.put(text)

    def close(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime abnormal behavior detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--model", type=str, default="dl_model/driver_behavior_classifier.keras")
    parser.add_argument("--labels", type=str, default="dl_model/driver_behavior_labels.json")
    parser.add_argument("--danger-threshold", type=float, default=0.65)
    parser.add_argument("--danger-confirm-frames", type=int, default=6)
    parser.add_argument("--voice-status", action="store_true")
    parser.add_argument("--voice-interval", type=float, default=5.0)
    return parser.parse_args()


def preprocess(frame_bgr: np.ndarray, img_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    return x[None, ...]


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = base / model_path
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = base / labels_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    model = tf.keras.models.load_model(str(model_path))
    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    announcer = VoiceAnnouncer(enabled=args.voice_status, interval=args.voice_interval)
    danger_streak = 0
    ema_probs = None
    alpha = 0.25
    last_t = time.perf_counter()
    fps_ema = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        now_t = time.perf_counter()
        dt = now_t - last_t
        last_t = now_t
        if not ok:
            break
        if dt > 0:
            fps_now = 1.0 / dt
            fps_ema = fps_now if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * fps_now)

        probs = model.predict(preprocess(frame, args.img_size), verbose=0)[0].astype(np.float32)
        ema_probs = probs if ema_probs is None else ((1.0 - alpha) * ema_probs + alpha * probs)
        pred_idx = int(np.argmax(ema_probs))
        pred_label = class_names[pred_idx]
        pred_prob = float(ema_probs[pred_idx])

        danger = pred_label.lower() in {"drunk", "smoking", "phone"} and pred_prob >= args.danger_threshold
        if danger:
            danger_streak += 1
        else:
            danger_streak = max(0, danger_streak - 1)
        alert_on = danger_streak >= max(1, args.danger_confirm_frames)

        color = (0, 255, 0)
        msg = "Normal"
        if alert_on:
            color = (0, 0, 255)
            msg = f"Abnormal: {pred_label}"
            if pred_label.lower() == "drunk":
                announcer.say("drunk", "Warning. Driver appears drunk.")
            elif pred_label.lower() == "smoking":
                announcer.say("smoking", "Warning. Smoking detected while driving.")
            elif pred_label.lower() == "phone":
                announcer.say("phone", "Warning. Phone usage detected while driving.")

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 95), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(frame, f"State: {msg}", (10, 30), font, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {pred_label} ({pred_prob:.2f})", (10, 58), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"DangerStreak: {danger_streak}  FPS: {fps_ema:.1f}", (10, 84), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        if alert_on:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 3)

        cv2.imshow("Driver Behavior Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    announcer.close()


if __name__ == "__main__":
    main()
