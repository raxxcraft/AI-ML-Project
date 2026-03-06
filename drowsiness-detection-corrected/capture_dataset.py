import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture real eye images for drowsiness training")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--out-dir", type=str, default="real_dataset", help="Output dataset directory")
    parser.add_argument("--img-size", type=int, default=32, help="Saved eye image size")
    parser.add_argument("--save-interval", type=float, default=0.08, help="Seconds between saved samples")
    parser.add_argument(
        "--class-label",
        type=str,
        default="all",
        choices=["all", "open", "closed"],
        help="Lock capture to one class for balanced data collection.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Stop automatically when this many images are saved for the active class (0 disables).",
    )
    parser.add_argument(
        "--start-collecting",
        action="store_true",
        help="Start capture immediately without pressing space.",
    )
    return parser.parse_args()


def pick_largest(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def heuristic_eye_boxes(face_w: int, face_h: int):
    # Face-relative fallback windows for left/right eye regions.
    eye_w = max(18, int(face_w * 0.28))
    eye_h = max(12, int(face_h * 0.20))
    eye_y = max(2, int(face_h * 0.20))
    left_x = max(2, int(face_w * 0.18))
    right_x = max(2, int(face_w * 0.54))
    return [(left_x, eye_y, eye_w, eye_h), (right_x, eye_y, eye_w, eye_h)]


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    out_root = (base / args.out_dir).resolve()
    closed_dir = out_root / "Closed"
    open_dir = out_root / "Open"
    closed_dir.mkdir(parents=True, exist_ok=True)
    open_dir.mkdir(parents=True, exist_ok=True)

    face = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_alt.xml"))
    left_eye = cv2.CascadeClassifier(str(base / "haarcascade_lefteye_2splits.xml"))
    right_eye = cv2.CascadeClassifier(str(base / "haarcascade_righteye_2splits.xml"))
    fallback_eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    if face.empty() or left_eye.empty() or right_eye.empty() or fallback_eye.empty():
        raise RuntimeError("Failed to load one or more Haar cascades.")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    mode = "Open"
    if args.class_label == "closed":
        mode = "Closed"
    if args.class_label == "open":
        mode = "Open"

    collecting = args.start_collecting
    save_index = 0
    last_save = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    if args.class_label == "all":
        print("Controls: [o]=Open [c]=Closed [space]=start/stop [s]=snapshot [q]=quit")
    else:
        print("Controls: [space]=start/stop [s]=snapshot [q]=quit")
        print(f"Class lock active: {mode}")
    print(f"Saving to: {out_root}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        best_face = pick_largest(faces)
        eye_boxes = []
        face_roi = None
        fx = fy = 0
        used_fallback_boxes = False
        snapshot = False

        if best_face is not None:
            fx, fy, fw, fh = best_face
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 220, 255), 2)
            upper_y_end = fy + int(0.65 * fh)
            face_roi = frame[fy:upper_y_end, fx : fx + fw]
            face_roi_gray = gray[fy:upper_y_end, fx : fx + fw]

            l_boxes = left_eye.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(14, 14))
            r_boxes = right_eye.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=2, minSize=(14, 14))
            l_box = pick_largest(l_boxes)
            r_box = pick_largest(r_boxes)

            if l_box is None or r_box is None:
                g_boxes = fallback_eye.detectMultiScale(
                    face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(14, 14)
                )
                g_sorted = sorted(g_boxes, key=lambda b: b[2] * b[3], reverse=True)
                if len(g_sorted) >= 2:
                    two = sorted(g_sorted[:2], key=lambda b: b[0])
                    if l_box is None:
                        l_box = two[0]
                    if r_box is None:
                        r_box = two[1]

            if l_box is not None:
                lx, ly, lw, lh = l_box
                eye_boxes.append((lx, ly, lw, lh))
                cv2.rectangle(frame, (fx + lx, fy + ly), (fx + lx + lw, fy + ly + lh), (0, 255, 0), 2)
            if r_box is not None:
                rx, ry, rw, rh = r_box
                eye_boxes.append((rx, ry, rw, rh))
                cv2.rectangle(frame, (fx + rx, fy + ry), (fx + rx + rw, fy + ry + rh), (255, 0, 0), 2)

            if len(eye_boxes) == 0:
                # Haar can miss eyes frequently; fallback keeps dataset capture usable.
                fallback_boxes = heuristic_eye_boxes(face_roi.shape[1], face_roi.shape[0])
                eye_boxes.extend(fallback_boxes)
                used_fallback_boxes = True
                for ex, ey, ew, eh in fallback_boxes:
                    cv2.rectangle(frame, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (0, 165, 255), 2)

        now = time.time()
        if collecting and face_roi is not None and len(eye_boxes) > 0 and (now - last_save) >= args.save_interval:
            target_dir = open_dir if mode == "Open" else closed_dir
            for x, y, w, h in eye_boxes:
                crop = face_roi[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
                out_file = target_dir / f"{mode.lower()}_{save_index:06d}.png"
                cv2.imwrite(str(out_file), crop)
                save_index += 1
            last_save = now

        closed_count = len(list(closed_dir.glob("*.png")))
        open_count = len(list(open_dir.glob("*.png")))
        if args.max_images > 0:
            active_count = open_count if mode == "Open" else closed_count
            if active_count >= args.max_images:
                print(f"Reached max-images target for {mode}: {active_count}")
                break

        state = "ON" if collecting else "OFF"
        cv2.putText(frame, f"Mode: {mode} | Capture: {state}", (10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Closed: {closed_count}  Open: {open_count}",
            (10, 50),
            font,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "o=open  c=closed  space=start/stop  q=quit",
            (10, 75),
            font,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Eyes: {len(eye_boxes)} {'(fallback)' if used_fallback_boxes else ''}",
            (10, 98),
            font,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Dataset Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("o") and args.class_label == "all":
            mode = "Open"
        if key == ord("c") and args.class_label == "all":
            mode = "Closed"
        if key == ord(" "):
            collecting = not collecting
        if key == ord("s"):
            snapshot = True

        if snapshot and face_roi is not None and len(eye_boxes) > 0:
            target_dir = open_dir if mode == "Open" else closed_dir
            for x, y, w, h in eye_boxes:
                crop = face_roi[y : y + h, x : x + w]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
                out_file = target_dir / f"{mode.lower()}_{save_index:06d}.png"
                cv2.imwrite(str(out_file), crop)
                save_index += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Dataset folder: {out_root}")


if __name__ == "__main__":
    main()
