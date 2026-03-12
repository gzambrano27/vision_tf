import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree as ET

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf
import keras

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
XML_DIR = DATA_DIR / 'annotations' / 'xml'
JSON_DIR = DATA_DIR / 'annotations' / 'json'
EXPORT_DIR = DATA_DIR / 'exports'
MODELS_DIR = DATA_DIR / 'models'
PREDICTIONS_DIR = DATA_DIR / 'predictions'
INDEX_FILE = DATA_DIR / 'dataset_index.json'
STATUS_FILE = DATA_DIR / 'training_status.json'
CLASSES_FILE = DATA_DIR / 'classes.json'
MODEL_META_FILE = MODELS_DIR / 'model_meta.json'

for path in [UPLOAD_DIR, XML_DIR, JSON_DIR, EXPORT_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 512


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return default


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def ensure_status_file():
    if not STATUS_FILE.exists():
        save_json(STATUS_FILE, {'status': 'idle', 'updated_at': utc_now(), 'progress': 0})


def load_index() -> List[Dict[str, Any]]:
    return load_json(INDEX_FILE, [])


def save_index(rows: List[Dict[str, Any]]):
    save_json(INDEX_FILE, rows)


def load_classes() -> List[str]:
    return load_json(CLASSES_FILE, [])


def save_classes(classes: List[str]):
    cleaned = []
    seen = set()
    for c in classes:
        c = str(c).strip()
        if c and c not in seen:
            cleaned.append(c)
            seen.add(c)
    save_json(CLASSES_FILE, cleaned)


def update_record(filename: str, width: int, height: int, depth: int, objects=None):
    rows = load_index()
    found = False
    thumb_url = f'/uploads/{filename}'
    for row in rows:
        if row['filename'] == filename:
            row['width'] = width
            row['height'] = height
            row['depth'] = depth
            row['thumbnail_url'] = thumb_url
            if objects is not None:
                row['objects'] = objects
            row['updated_at'] = utc_now()
            found = True
            break
    if not found:
        rows.append({
            'filename': filename,
            'width': width,
            'height': height,
            'depth': depth,
            'thumbnail_url': thumb_url,
            'objects': objects or [],
            'created_at': utc_now(),
            'updated_at': utc_now(),
        })
    save_index(rows)


def normalize_objects(objects: List[Dict[str, Any]], width: int, height: int):
    normalized = []
    for raw in objects:
        label = str(raw.get('label', '')).strip()
        if not label:
            continue
        xmin = max(0, min(width, int(raw.get('xmin', 0))))
        ymin = max(0, min(height, int(raw.get('ymin', 0))))
        xmax = max(0, min(width, int(raw.get('xmax', 0))))
        ymax = max(0, min(height, int(raw.get('ymax', 0))))
        if xmax <= xmin or ymax <= ymin:
            continue
        normalized.append({
            'id': raw.get('id') or f"{label}_{xmin}_{ymin}_{xmax}_{ymax}",
            'label': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
        })
    return normalized


def save_pascal_voc_xml(filename: str, width: int, height: int, depth: int, objects: List[Dict[str, Any]]):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'uploads'
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = str((UPLOAD_DIR / filename).resolve())
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Vision Trainer Studio TF'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(annotation, 'segmented').text = '0'
    for obj in objects:
        obj_el = ET.SubElement(annotation, 'object')
        ET.SubElement(obj_el, 'name').text = obj['label']
        ET.SubElement(obj_el, 'pose').text = 'Unspecified'
        ET.SubElement(obj_el, 'truncated').text = '0'
        ET.SubElement(obj_el, 'difficult').text = '0'
        bbox = ET.SubElement(obj_el, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(int(obj['xmin']))
        ET.SubElement(bbox, 'ymin').text = str(int(obj['ymin']))
        ET.SubElement(bbox, 'xmax').text = str(int(obj['xmax']))
        ET.SubElement(bbox, 'ymax').text = str(int(obj['ymax']))
    tree = ET.ElementTree(annotation)
    tree.write(XML_DIR / f'{Path(filename).stem}.xml', encoding='utf-8', xml_declaration=True)


def save_annotation_json(filename: str, width: int, height: int, depth: int, objects: List[Dict[str, Any]]):
    save_json(JSON_DIR / f'{Path(filename).stem}.json', {
        'filename': filename,
        'width': width,
        'height': height,
        'depth': depth,
        'objects': objects,
        'updated_at': utc_now(),
    })


def prepare_export() -> Dict[str, Any]:
    rows = load_index()
    classes = load_classes()
    annotated = [row for row in rows if row.get('objects')]
    export_dir = EXPORT_DIR / 'tensorflow_dataset'
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_rows = []
    for row in annotated:
        export_rows.append({
            'image_path': str((UPLOAD_DIR / row['filename']).resolve()),
            'filename': row['filename'],
            'width': row['width'],
            'height': row['height'],
            'depth': row['depth'],
            'objects': row['objects'],
        })
    save_json(export_dir / 'annotations.json', export_rows)
    meta = {
        'classes': classes,
        'num_images': len(rows),
        'num_annotated_images': len(annotated),
        'generated_at': utc_now(),
        'export_dir': str(export_dir),
        'annotation_json': str(export_dir / 'annotations.json'),
        'formats': ['Pascal VOC XML', 'JSON'],
    }
    save_json(export_dir / 'dataset_meta.json', meta)
    return meta


@app.route('/')
def index():
    ensure_status_file()
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/predictions/<path:filename>')
def serve_prediction(filename):
    return send_from_directory(PREDICTIONS_DIR, filename)


@app.route('/api/images', methods=['GET'])
def api_images():
    return jsonify(load_index())


@app.route('/api/classes', methods=['GET'])
def api_classes_get():
    return jsonify({'classes': load_classes()})


@app.route('/api/classes', methods=['POST'])
def api_classes_post():
    payload = request.get_json(force=True)
    save_classes(payload.get('classes', []))
    return jsonify({'ok': True, 'classes': load_classes()})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    files = request.files.getlist('images')
    uploaded = []
    if not files:
        return jsonify({'ok': False, 'error': 'No se recibieron imágenes.'}), 400
    for file in files:
        if not file.filename or not allowed_file(file.filename):
            continue
        filename = secure_filename(file.filename)
        target = UPLOAD_DIR / filename
        file.save(target)
        with Image.open(target) as img:
            width, height = img.size
            depth = len(img.getbands())
        update_record(filename, width, height, depth)
        uploaded.append({'filename': filename, 'width': width, 'height': height, 'depth': depth})
    return jsonify({'ok': True, 'uploaded': uploaded})


@app.route('/api/annotation/<filename>', methods=['GET'])
def api_get_annotation(filename):
    rows = load_index()
    for row in rows:
        if row['filename'] == filename:
            return jsonify({'ok': True, 'image': row})
    return jsonify({'ok': False, 'error': 'Imagen no encontrada.'}), 404


@app.route('/api/annotation/<filename>', methods=['POST'])
def api_save_annotation(filename):
    image_path = UPLOAD_DIR / filename
    if not image_path.exists():
        return jsonify({'ok': False, 'error': 'Imagen no encontrada.'}), 404
    payload = request.get_json(force=True)
    with Image.open(image_path) as img:
        width, height = img.size
        depth = len(img.getbands())
    objects = normalize_objects(payload.get('objects', []), width, height)
    update_record(filename, width, height, depth, objects)
    save_pascal_voc_xml(filename, width, height, depth, objects)
    save_annotation_json(filename, width, height, depth, objects)
    all_labels = sorted({obj['label'] for row in load_index() for obj in row.get('objects', []) if obj.get('label')})
    save_classes(sorted(set(load_classes()) | set(all_labels)))
    return jsonify({
        'ok': True,
        'xml_file': f'{Path(filename).stem}.xml',
        'json_file': f'{Path(filename).stem}.json',
        'classes': load_classes(),
        'objects': objects,
    })


@app.route('/api/export', methods=['GET'])
def api_export():
    meta = prepare_export()
    return jsonify({'ok': True, 'meta': meta})


@app.route('/api/train', methods=['POST'])
def api_train():
    ensure_status_file()
    payload = request.get_json(silent=True) or {}
    epochs = int(payload.get('epochs', 30))
    imgsz = int(payload.get('imgsz', 416))
    batch = int(payload.get('batch', 8))
    grid_size = int(payload.get('grid_size', 10))
    backbone = str(payload.get('backbone', 'MobileNetV2'))
    val_ratio = float(payload.get('val_ratio', 0.2))

    meta = prepare_export()
    if meta['num_annotated_images'] < 5:
        return jsonify({'ok': False, 'error': 'Necesitas al menos 5 imágenes anotadas para un entrenamiento mínimamente útil.'}), 400
    if len(meta['classes']) < 1:
        return jsonify({'ok': False, 'error': 'No hay clases definidas.'}), 400

    status = load_json(STATUS_FILE, {})
    if status.get('status') == 'running':
        return jsonify({'ok': False, 'error': 'Ya hay un entrenamiento en curso.'}), 400

    save_json(STATUS_FILE, {
        'status': 'running',
        'message': 'Iniciando entrenamiento TensorFlow...',
        'updated_at': utc_now(),
        'progress': 3,
        'config': {
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'grid_size': grid_size,
            'backbone': backbone,
            'val_ratio': val_ratio,
        },
    })

    cmd = [
        sys.executable,
        str(BASE_DIR / 'train_worker.py'),
        '--annotations', str(EXPORT_DIR / 'tensorflow_dataset' / 'annotations.json'),
        '--classes', str(CLASSES_FILE),
        '--epochs', str(epochs),
        '--imgsz', str(imgsz),
        '--batch', str(batch),
        '--grid-size', str(grid_size),
        '--backbone', backbone,
        '--val-ratio', str(val_ratio),
        '--status-file', str(STATUS_FILE),
        '--models-dir', str(MODELS_DIR),
    ]
    subprocess.Popen(cmd, cwd=BASE_DIR)
    return jsonify({'ok': True, 'meta': meta, 'message': 'Entrenamiento TensorFlow iniciado.'})


@app.route('/api/train/status', methods=['GET'])
def api_train_status():
    ensure_status_file()
    return jsonify(load_json(STATUS_FILE, {'status': 'idle'}))


# Registramos la función globalmente para que Keras pueda reconocerla al deserializar
@keras.saving.register_keras_serializable()
def split_heads(t):
    obj = t[..., 0:1]
    box = tf.sigmoid(t[..., 1:5])
    cls = t[..., 5:]
    return tf.concat([obj, box, cls], axis=-1)

_MODEL_CACHE = {'path': None, 'model': None, 'meta': None}

def get_model_and_meta():
    status = load_json(STATUS_FILE, {})
    model_path = status.get('best_model')
    if not model_path or not Path(model_path).exists() or not MODEL_META_FILE.exists():
        return None, None
    if _MODEL_CACHE['path'] == model_path and _MODEL_CACHE['model'] is not None:
        return _MODEL_CACHE['model'], _MODEL_CACHE['meta']
    
    # Cargamos el modelo
    model = tf.keras.models.load_model(model_path, compile=False)
    meta = load_json(MODEL_META_FILE, {})
    _MODEL_CACHE.update({'path': model_path, 'model': model, 'meta': meta})
    return model, meta


@app.route('/api/predict', methods=['POST'])
def api_predict():
    file = request.files.get('image')
    conf = float(request.form.get('conf', 0.25))
    if not file or not file.filename:
        return jsonify({'ok': False, 'error': 'No se recibió imagen para inferencia.'}), 400
    model, meta = get_model_and_meta()
    if model is None:
        return jsonify({'ok': False, 'error': 'No existe un modelo TensorFlow entrenado disponible todavía.'}), 400

    import numpy as np

    filename = secure_filename(file.filename)
    temp_input = PREDICTIONS_DIR / f'input_{filename}'
    file.save(temp_input)

    img = Image.open(temp_input).convert('RGB')
    original_w, original_h = img.size
    imgsz = int(meta['imgsz'])
    grid = int(meta['grid_size'])
    classes = meta['classes']
    arr = np.asarray(img.resize((imgsz, imgsz)), dtype='float32') / 255.0
    pred = model.predict(arr[None, ...], verbose=0)[0]

    objectness = 1.0 / (1.0 + np.exp(-pred[..., 0]))
    bbox = 1.0 / (1.0 + np.exp(-pred[..., 1:5]))
    class_logits = pred[..., 5:]
    class_probs = tf.nn.softmax(class_logits, axis=-1).numpy()

    boxes = []
    scores = []
    labels = []
    cell_w = original_w / grid
    cell_h = original_h / grid
    for gy in range(grid):
        for gx in range(grid):
            obj_score = float(objectness[gy, gx])
            cls_idx = int(np.argmax(class_probs[gy, gx]))
            cls_score = float(class_probs[gy, gx, cls_idx])
            score = obj_score * cls_score
            if score < conf:
                continue
            cx = (gx + bbox[gy, gx, 0]) * cell_w
            cy = (gy + bbox[gy, gx, 1]) * cell_h
            bw = bbox[gy, gx, 2] * original_w
            bh = bbox[gy, gx, 3] * original_h
            xmin = max(0.0, cx - bw / 2.0)
            ymin = max(0.0, cy - bh / 2.0)
            xmax = min(float(original_w), cx + bw / 2.0)
            ymax = min(float(original_h), cy + bh / 2.0)
            boxes.append([ymin / original_h, xmin / original_w, ymax / original_h, xmax / original_w])
            scores.append(score)
            labels.append(classes[cls_idx])

    detections = []
    preview = img.copy()
    if boxes:
        selected = tf.image.non_max_suppression(
            boxes=np.array([[b[1]*original_w, b[0]*original_h, b[3]*original_w, b[2]*original_h] for b in boxes], dtype=np.float32),
            scores=np.array(scores, dtype=np.float32),
            max_output_size=50,
            iou_threshold=0.45,
            score_threshold=conf,
        ).numpy().tolist()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(preview)
        for idx in selected:
            ymin, xmin, ymax, xmax = boxes[idx]
            x1, y1, x2, y2 = xmin * original_w, ymin * original_h, xmax * original_w, ymax * original_h
            draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=3)
            label = f"{labels[idx]} {scores[idx]:.2f}"
            draw.text((x1 + 4, max(4, y1 - 16)), label, fill=(34, 197, 94))
            detections.append({
                'label': labels[idx],
                'confidence': round(float(scores[idx]), 4),
                'xmin': round(float(x1), 2),
                'ymin': round(float(y1), 2),
                'xmax': round(float(x2), 2),
                'ymax': round(float(y2), 2),
            })
    output_name = f'pred_{Path(filename).stem}.jpg'
    output_path = PREDICTIONS_DIR / output_name
    preview.save(output_path)
    return jsonify({'ok': True, 'detections': detections, 'preview_url': f'/predictions/{output_name}'})


@app.route('/api/download/model', methods=['GET'])
def api_download_model():
    status = load_json(STATUS_FILE, {})
    best_model = status.get('best_model')
    if not best_model or not Path(best_model).exists():
        return jsonify({'ok': False, 'error': 'Aún no hay modelo entrenado para descargar.'}), 404
    return send_file(best_model, as_attachment=True)


if __name__ == '__main__':
    ensure_status_file()
    app.run(debug=True)