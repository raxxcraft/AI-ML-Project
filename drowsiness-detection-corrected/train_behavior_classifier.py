import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-class abnormal driving behavior classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset root with subfolders: normal, drunk, smoking, phone",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--output-model",
        type=str,
        default="dl_model/driver_behavior_classifier.keras",
        help="Output model path",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default="dl_model/driver_behavior_labels.json",
        help="Output labels JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # Prefer common valid formats and skip unknown/corrupt files at the decoder level.
    train_ds = image_dataset_from_directory(
        str(data_dir),
        validation_split=args.val_split,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    class_names = train_ds.class_names

    val_ds = image_dataset_from_directory(
        str(data_dir),
        validation_split=args.val_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes in dataset.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1024).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    data_aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ]
    )

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = data_aug(x)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(args.img_size, args.img_size, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    out_model = Path(args.output_model)
    if not out_model.is_absolute():
        out_model = base / out_model
    out_model.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_model if out_model.suffix.lower() in {".h5", ".hdf5"} else out_model.with_suffix(".h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Light fine-tuning
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=max(3, args.epochs // 3), callbacks=callbacks)

    best_model = tf.keras.models.load_model(str(checkpoint_path))
    best_model.save(str(out_model))
    print(f"Saved model: {out_model.resolve()}")

    out_labels = Path(args.output_labels)
    if not out_labels.is_absolute():
        out_labels = base / out_labels
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved labels: {out_labels.resolve()}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
