import os, json, argparse, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def build_model(num_classes, img_size):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False  # Phase 1: freeze
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base

def plot_confusion_matrix(cm, classes, out_path):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Root with train/val/(test)")
    parser.add_argument("--img", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs1", type=int, default=10, help="Epochs (frozen)")
    parser.add_argument("--epochs2", type=int, default=10, help="Epochs (fine-tune)")
    args = parser.parse_args()

    # GPU memory growth (optional)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] Using GPU")
        except Exception as e:
            print("[WARN] Could not set memory growth:", e)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    # Data generators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2]
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(args.img, args.img),
        batch_size=args.batch,
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(args.img, args.img),
        batch_size=args.batch,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    id_to_label = {v: k for k, v in class_indices.items()}
    os.makedirs("models", exist_ok=True)
    with open("models/label_map.json", "w") as f:
        json.dump(id_to_label, f, indent=2)
    print("[INFO] Classes:", id_to_label)

    # Build + compile
    model, base = build_model(num_classes, args.img)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/best_phase1.keras", monitor="val_accuracy", save_best_only=True)
    ]

    # Phase 1: train top
    history1 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs1, callbacks=cbs)

    # Phase 2: fine-tune some layers
    base.trainable = True
    # unfreeze last 50 layers
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    cbs2 = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint("models/best_finetuned.keras", monitor="val_accuracy", save_best_only=True)
    ]
    history2 = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs2, callbacks=cbs2)

    # Save final
    model.save("models/egg_model.keras")
    print("[INFO] Saved model to models/egg_model.keras")

    # Evaluate on val and test (if available)
    def evaluate(gen, name):
        preds = model.predict(gen)
        y_pred = np.argmax(preds, axis=1)
        y_true = gen.classes
        labels = [id_to_label[i] for i in range(num_classes)]
        report = classification_report(y_true, y_pred, target_names=labels, digits=4)
        print(f"\n===== {name} Classification Report =====\n{report}")
        with open(f"models/classification_report_{name}.txt", "w") as f:
            f.write(report)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, labels, f"models/confusion_matrix_{name}.png")

    evaluate(val_gen, "val")

    if os.path.isdir(test_dir) and any(os.scandir(test_dir)):
        test_gen = val_datagen.flow_from_directory(
            test_dir,
            target_size=(args.img, args.img),
            batch_size=args.batch,
            class_mode="categorical",
            shuffle=False
        )
        evaluate(test_gen, "test")
    else:
        print("[INFO] No test set found. Skipping test evaluation.")

if __name__ == "__main__":
    main()
