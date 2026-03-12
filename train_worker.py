import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def utc_now():
    return datetime.utcnow().isoformat()


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default


def update_status(path: Path, **kwargs):
    current = load_json(path, {})
    current.update(kwargs)
    current['updated_at'] = utc_now()
    save_json(path, current)


def resize_bbox(obj, old_w, old_h, new_size):
    sx = new_size / old_w
    sy = new_size / old_h
    return {
        'label': obj['label'],
        'xmin': obj['xmin'] * sx,
        'ymin': obj['ymin'] * sy,
        'xmax': obj['xmax'] * sx,
        'ymax': obj['ymax'] * sy,
    }


def encode_targets(objects, classes, img_size, grid_size):
    class_map = {c: i for i, c in enumerate(classes)}
    target = np.zeros((grid_size, grid_size, 5 + len(classes)), dtype='float32')
    cell_size = img_size / grid_size
    collisions = 0
    for obj in objects:
        if obj['label'] not in class_map:
            continue
        cx = (obj['xmin'] + obj['xmax']) / 2.0
        cy = (obj['ymin'] + obj['ymax']) / 2.0
        bw = max(1.0, obj['xmax'] - obj['xmin']) / img_size
        bh = max(1.0, obj['ymax'] - obj['ymin']) / img_size
        gx = min(grid_size - 1, max(0, int(cx / cell_size)))
        gy = min(grid_size - 1, max(0, int(cy / cell_size)))
        if target[gy, gx, 0] == 1.0:
            collisions += 1
            continue
        rel_x = (cx / cell_size) - gx
        rel_y = (cy / cell_size) - gy
        target[gy, gx, 0] = 1.0
        target[gy, gx, 1] = np.clip(rel_x, 0.0, 1.0)
        target[gy, gx, 2] = np.clip(rel_y, 0.0, 1.0)
        target[gy, gx, 3] = np.clip(bw, 0.0, 1.0)
        target[gy, gx, 4] = np.clip(bh, 0.0, 1.0)
        target[gy, gx, 5 + class_map[obj['label']]] = 1.0
    return target, collisions


def build_dataset(annotations, classes, img_size, grid_size):
    images = []
    targets = []
    collisions_total = 0
    for row in annotations:
        img = Image.open(row['image_path']).convert('RGB')
        old_w, old_h = img.size
        img = img.resize((img_size, img_size))
        arr = np.asarray(img, dtype='float32') / 255.0
        resized_objects = [resize_bbox(obj, old_w, old_h, img_size) for obj in row.get('objects', [])]
        target, collisions = encode_targets(resized_objects, classes, img_size, grid_size)
        collisions_total += collisions
        images.append(arr)
        targets.append(target)
    return np.array(images, dtype='float32'), np.array(targets, dtype='float32'), collisions_total


def split_data(annotations, val_ratio):
    rnd = random.Random(42)
    items = annotations[:]
    rnd.shuffle(items)
    val_count = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    val = items[:val_count]
    train = items[val_count:] if val_count < len(items) else items
    if not train:
        train, val = items, []
    return train, val


def make_model(img_size, grid_size, num_classes, backbone_name='MobileNetV2'):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name='image')
    if backbone_name == 'EfficientNetB0':
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_tensor=inputs,
            weights='imagenet',
        )
    else:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            input_tensor=inputs,
            weights='imagenet',
        )
    backbone.trainable = True
    x = backbone.output
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D((5 + num_classes), 1, padding='same')(x)
    x = tf.keras.layers.Resizing(grid_size, grid_size, interpolation='bilinear')(x)

    def split_heads(t):
        obj = t[..., 0:1]              # logits
        box = tf.sigmoid(t[..., 1:5])  # 0..1
        cls = t[..., 5:]               # logits
        return tf.concat([obj, box, cls], axis=-1)

    outputs = tf.keras.layers.Lambda(split_heads, name='detector_output')(x)
    return tf.keras.Model(inputs, outputs, name='grid_detector_tf')


def detection_loss(num_classes):
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        obj_true = y_true[..., 0]          # [B, G, G]
        box_true = y_true[..., 1:5]        # [B, G, G, 4]
        cls_true = y_true[..., 5:]         # [B, G, G, C]

        obj_logits = y_pred[..., 0]        # [B, G, G]
        box_pred = y_pred[..., 1:5]        # [B, G, G, 4]
        cls_logits = y_pred[..., 5:]       # [B, G, G, C]

        # 1) objectness loss -> [B, G, G]
        obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=obj_true,
            logits=obj_logits,
        )

        # 2) box loss -> [B, G, G]
        # solo penaliza celdas con objeto
        box_loss_per_coord = tf.keras.losses.huber(
            box_true,
            box_pred,
        )  # Ya sale con forma [B, G, G] porque Keras reduce la última dimensión
        box_loss = box_loss_per_coord * obj_true

        # 3) class loss -> [B, G, G]
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=cls_true,
            logits=cls_logits,
        ) * obj_true

        total = obj_loss + box_loss + cls_loss
        return tf.reduce_mean(total)

    return loss_fn


class StatusCallback:
    def __init__(self, status_file: Path, total_epochs: int):
        self.status_file = status_file
        self.total_epochs = total_epochs

    def __call__(self):
        import tensorflow as tf
        status_file = self.status_file
        total_epochs = self.total_epochs

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                progress = int(((epoch + 1) / max(1, total_epochs)) * 100)
                update_status(
                    status_file,
                    status='running',
                    progress=progress,
                    message=(
                        f"Época {epoch + 1}/{total_epochs} | "
                        f"loss={logs.get('loss', 0):.4f} | "
                        f"val_loss={logs.get('val_loss', 0):.4f}"
                    ),
                    metrics={k: float(v) for k, v in logs.items() if isinstance(v, (int, float))},
                )

        return _CB()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--grid-size', type=int, default=10)
    parser.add_argument('--backbone', default='MobileNetV2')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--status-file', required=True)
    parser.add_argument('--models-dir', required=True)
    args = parser.parse_args()

    status_file = Path(args.status_file)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    update_status(status_file, status='running', progress=5, message='Cargando TensorFlow y preparando dataset...')

    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    annotations = load_json(Path(args.annotations), [])
    classes = load_json(Path(args.classes), [])
    if not annotations:
        raise ValueError('No hay anotaciones exportadas para entrenar.')
    if not classes:
        raise ValueError('No hay clases definidas.')

    train_rows, val_rows = split_data(annotations, args.val_ratio)
    x_train, y_train, collisions_train = build_dataset(train_rows, classes, args.imgsz, args.grid_size)
    x_val, y_val, collisions_val = build_dataset(val_rows, classes, args.imgsz, args.grid_size) if val_rows else (None, None, 0)

    update_status(
        status_file,
        status='running',
        progress=12,
        message=(
            f'Dataset listo. Train={len(x_train)} | Val={0 if x_val is None else len(x_val)} | '
            f'Colisiones de celda omitidas={collisions_train + collisions_val}'
        ),
    )

    model = make_model(args.imgsz, args.grid_size, len(classes), args.backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=detection_loss(len(classes)),
    )

    callbacks = [
        StatusCallback(status_file, args.epochs)(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / 'best.keras'),
            monitor='val_loss' if x_val is not None else 'loss',
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if x_val is not None else 'loss',
            patience=6,
            restore_best_weights=True,
        ),
    ]

    fit_kwargs = dict(
        x=x_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )
    if x_val is not None and len(x_val) > 0:
        fit_kwargs['validation_data'] = (x_val, y_val)

    history = model.fit(**fit_kwargs)
    final_model_path = models_dir / 'best.keras'
    model.save(final_model_path)

    model_meta = {
        'classes': classes,
        'imgsz': args.imgsz,
        'grid_size': args.grid_size,
        'backbone': args.backbone,
        'epochs_requested': args.epochs,
        'epochs_ran': len(history.history.get('loss', [])),
        'train_samples': len(x_train),
        'val_samples': 0 if x_val is None else len(x_val),
        'collisions_ignored': collisions_train + collisions_val,
        'updated_at': utc_now(),
    }
    save_json(models_dir / 'model_meta.json', model_meta)

    metrics = {k: [float(vv) for vv in vals] for k, vals in history.history.items()}
    save_json(models_dir / 'history.json', metrics)

    update_status(
        status_file,
        status='completed',
        progress=100,
        message='Entrenamiento TensorFlow completado correctamente.',
        best_model=str(final_model_path),
        model_meta=model_meta,
        final_metrics={k: v[-1] for k, v in metrics.items() if v},
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Try to salvage status file path from argv.
        import sys
        status_path = None
        if '--status-file' in sys.argv:
            idx = sys.argv.index('--status-file')
            if idx + 1 < len(sys.argv):
                status_path = Path(sys.argv[idx + 1])
        if status_path:
            update_status(status_path, status='error', progress=0, message=f'Error durante entrenamiento: {exc}')
        raise
