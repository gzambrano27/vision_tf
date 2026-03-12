# Vision Trainer Studio TF

Aplicación web local para:
- cargar imágenes,
- dibujar **múltiples cajas por imagen**,
- guardar anotaciones en **Pascal VOC XML** y JSON,
- exportar el dataset para entrenamiento,
- entrenar un detector **TensorFlow/Keras**,
- probar inferencia sobre imágenes nuevas,
- descargar el modelo entrenado en `.keras`.

## Requisitos
- Python **3.10, 3.11 o 3.12**
- Windows, Linux o macOS
- Recomendado: 8 GB de RAM o más

## Instalación en Windows PowerShell
Desde la carpeta del proyecto:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

Abre en el navegador:

```text
http://127.0.0.1:5000
```

## Flujo recomendado
1. Sube imágenes.
2. Crea tus clases: `carro`, `gato`, `casco`, etc.
3. Selecciona una imagen.
4. Elige la clase activa y dibuja todas las cajas que quieras en esa imagen.
5. Guarda la anotación.
6. Repite con varias imágenes.
7. Exporta el dataset.
8. Entrena el modelo TensorFlow.
9. Cuando termine, usa la sección de inferencia.
10. Descarga el modelo `.keras`.

## Qué cambió en esta versión
- Interfaz visual más limpia y moderna.
- Selección rápida de clases mediante chips.
- Vista del dataset con miniaturas.
- Múltiples anotaciones por imagen.
- Backend de entrenamiento migrado a **TensorFlow/Keras**.
- Modelo exportado en formato **`.keras`**.

## Notas importantes
- La precisión real depende del dataset; ningún software puede garantizar precisión alta si hay pocas imágenes o cajas mal hechas.
- Este detector usa una arquitectura tipo **grid detector** sobre un backbone preentrenado (`MobileNetV2` o `EfficientNetB0`) para mantener el proyecto ligero y fácil de instalar.
- Si necesitas más precisión en datasets grandes, esta base puede evolucionarse luego a RetinaNet, EfficientDet o KerasCV.

## Estructura
- `data/uploads/`: imágenes cargadas
- `data/annotations/xml/`: XML Pascal VOC
- `data/annotations/json/`: anotaciones JSON
- `data/exports/tensorflow_dataset/`: exportación del dataset
- `data/models/`: modelo `.keras`, historial y metadatos
- `data/predictions/`: resultados de inferencia
