import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced drowsiness CNN trainer")
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        required=True,
        help="Dataset root with subfolders Closed/ and Open/. Repeat to combine folders.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="auto",
        choices=["classic", "residual", "separable", "auto"],
        help="Training algorithm. 'auto' trains all and keeps the best model.",
    )
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-copy",
        action="store_true",
        help="Upsample minority class by copying files during merge.",
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable class-weight balancing during optimization.",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="_merged_dataset",
        help="Working folder where all input datasets are merged",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dl_model/drowsiness_cnn.h5",
        help="Output model path, relative to this script folder unless absolute",
    )
    return parser.parse_args()


def conv_block(filters):
    return [
        layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    ]


def sep_block(filters):
    return [
        layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
    ]


def build_model(algorithm: str, input_shape=(32, 32, 3)):
    if algorithm == "classic":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.35),
                layers.Dense(128, activation="relu"),
                layers.Dense(2, activation="softmax"),
            ]
        )
    elif algorithm == "separable":
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                *sep_block(32),
                *sep_block(64),
                *sep_block(128),
                layers.GlobalAveragePooling2D(),
                layers.Dense(160, activation="relu"),
                layers.Dropout(0.4),
                layers.Dense(2, activation="softmax"),
            ]
        )
    else:
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                *conv_block(32),
                *conv_block(64),
                *conv_block(128),
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.4),
                layers.Dense(2, activation="softmax"),
            ]
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def collect_images(dataset_roots):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    result = {"Closed": [], "Open": []}
    for root in dataset_roots:
        for cls in ("Closed", "Open"):
            cls_dir = root / cls
            if cls_dir.exists():
                files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
                result[cls].extend(files)
    return result


def build_merged_dataset(merged_dir: Path, image_map: dict, balance_copy: bool, seed: int):
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    (merged_dir / "Closed").mkdir(parents=True, exist_ok=True)
    (merged_dir / "Open").mkdir(parents=True, exist_ok=True)

    if balance_copy:
        rng = np.random.default_rng(seed)
        closed = list(image_map["Closed"])
        open_ = list(image_map["Open"])
        target = max(len(closed), len(open_))
        if len(closed) < target:
            closed.extend(list(rng.choice(closed, size=target - len(closed), replace=True)))
        if len(open_) < target:
            open_.extend(list(rng.choice(open_, size=target - len(open_), replace=True)))
        image_map = {"Closed": closed, "Open": open_}

    for cls in ("Closed", "Open"):
        for idx, src in enumerate(image_map[cls]):
            dst = merged_dir / cls / f"{cls.lower()}_{idx:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)


def make_generators(merged_dir: Path, args):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=args.val_split,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.05,
        zoom_range=0.10,
        brightness_range=(0.80, 1.20),
        horizontal_flip=False,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=args.val_split)

    train_ds = train_gen.flow_from_directory(
        str(merged_dir),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        classes=["Closed", "Open"],
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=args.seed,
    )
    val_ds = val_gen.flow_from_directory(
        str(merged_dir),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        classes=["Closed", "Open"],
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )
    return train_ds, val_ds


def compute_class_weight(train_ds):
    class_counts = np.bincount(train_ds.classes)
    total = int(class_counts.sum())
    class_weight = {}
    for cls_idx, count in enumerate(class_counts):
        class_weight[cls_idx] = float(total) / float(max(1, len(class_counts) * count))
    return class_weight


def evaluate_model(model, val_ds):
    val_ds.reset()
    probs = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = val_ds.classes

    conf = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    tn, fp, fn, tp = conf[1, 1], conf[1, 0], conf[0, 1], conf[0, 0]
    closed_precision = float(tp) / float(max(1, tp + fp))
    closed_recall = float(tp) / float(max(1, tp + fn))
    closed_f1 = (2.0 * closed_precision * closed_recall) / max(1e-8, closed_precision + closed_recall)
    val_acc = float(np.mean(y_true == y_pred))
    return {
        "val_acc": val_acc,
        "closed_precision": closed_precision,
        "closed_recall": closed_recall,
        "closed_f1": closed_f1,
        "confusion": conf,
    }


def train_one_algorithm(algo, train_ds, val_ds, class_weight, out_path, epochs):
    model = build_model(algorithm=algo, input_shape=(train_ds.target_size[0], train_ds.target_size[1], 3))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    train_ds.reset()
    val_ds.reset()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    best_val_acc = float(np.max(history.history.get("val_accuracy", [0.0])))
    best_model = tf.keras.models.load_model(str(out_path))
    metrics = evaluate_model(best_model, val_ds)
    return best_val_acc, metrics


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent
    dataset_roots = [Path(p).resolve() for p in args.data_dir]
    for root in dataset_roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset not found: {root}")

    image_map = collect_images(dataset_roots)
    closed_total = len(image_map["Closed"])
    open_total = len(image_map["Open"])
    print(f"Collected images -> Closed: {closed_total}, Open: {open_total}")
    if closed_total == 0 or open_total == 0:
        raise ValueError(
            "No training images found. Expected structure: <data-dir>/Closed and <data-dir>/Open with image files."
        )
    ratio = max(closed_total, open_total) / max(1, min(closed_total, open_total))
    if ratio > 1.5:
        print(
            f"Warning: class imbalance is high ({ratio:.2f}x). "
            "Collect more samples for minority class for better real-time detection."
        )

    merged_dir_arg = Path(args.merged_dir)
    merged_dir = merged_dir_arg if merged_dir_arg.is_absolute() else (base / merged_dir_arg)
    build_merged_dataset(merged_dir, image_map, balance_copy=args.balance_copy, seed=args.seed)
    print(f"Merged dataset at: {merged_dir}")

    train_ds, val_ds = make_generators(merged_dir, args)
    if train_ds.samples == 0 or val_ds.samples == 0:
        raise ValueError(
            "No training images found. Expected structure: <data-dir>/Closed and <data-dir>/Open with image files."
        )

    print(f"Class indices: {train_ds.class_indices}")
    if args.no_class_weight:
        class_weight = None
        print("Class weights: disabled")
    else:
        class_weight = compute_class_weight(train_ds)
        print(f"Class weights: {class_weight}")

    out_arg = Path(args.output)
    out_path = out_arg if out_arg.is_absolute() else (base / out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = ["classic", "residual", "separable"] if args.algorithm == "auto" else [args.algorithm]
    best_algo = None
    best_score = -1.0
    best_metrics = None
    best_model_path = None

    for algo in algorithms:
        candidate = out_path.with_name(f"{out_path.stem}_{algo}{out_path.suffix}")
        print(f"\n=== Training algorithm: {algo} ===")
        best_val_acc, metrics = train_one_algorithm(
            algo=algo,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weight=class_weight,
            out_path=candidate,
            epochs=args.epochs,
        )
        print(
            f"{algo} -> val_acc={metrics['val_acc']:.4f}, closed_f1={metrics['closed_f1']:.4f}, "
            f"closed_precision={metrics['closed_precision']:.4f}, closed_recall={metrics['closed_recall']:.4f}"
        )
        print(f"{algo} confusion matrix [[Closed->Closed, Closed->Open],[Open->Closed, Open->Open]]:")
        print(metrics["confusion"])

        score = (0.7 * best_val_acc) + (0.3 * metrics["closed_f1"])
        if score > best_score:
            best_score = score
            best_algo = algo
            best_metrics = metrics
            best_model_path = candidate

    shutil.copy2(best_model_path, out_path)
    print(f"\nSelected best algorithm: {best_algo}")
    print(
        f"Best metrics -> val_acc={best_metrics['val_acc']:.4f}, closed_f1={best_metrics['closed_f1']:.4f}, "
        f"closed_precision={best_metrics['closed_precision']:.4f}, closed_recall={best_metrics['closed_recall']:.4f}"
    )
    print(f"Saved best model to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
