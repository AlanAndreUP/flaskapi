import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from PIL import Image
import io

MODEL_PATH = 'mi_modelo_fatiga.h5' 
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 24
IMAGE_CHANNELS = 1
COLOR_MODE = 'L' if IMAGE_CHANNELS == 1 else 'RGB'
CLASS_NAMES = [
    "Estado Normal (ojos abiertos)",
    "Fatiga Leve (parpadeo frecuente, ojos ligeramente entrecerrados)",
    "Fatiga Moderada (parpadeo excesivo, ojos entrecerrados, inclinación de cabeza)",
    "Fatiga Severa (ojos casi cerrados, cabeza caída, señales de somnolencia)",
    "Ojos secos (irritación ocular visible)",
    "Visión borrosa (expresión de esfuerzo, ojos entrecerrados al enfocar)",
    "Frotado de ojos (movimientos de frotado con las manos)",
    "Desviación de mirada (mirada perdida, falta de enfoque en pantalla)",
    "Dolor de cabeza (expresión de incomodidad, ceño fruncido)",
    "Enrojecimiento ocular (ojos notablemente irritados)",
    "Cierre involuntario de ojos (micro-siestas, caída de párpados sin control)",
    "Postura incorrecta por fatiga (inclinación de cabeza, cuello tenso)",
    "Sobreesfuerzo visual (uso excesivo de la vista, ojos muy abiertos o tensión ocular)"
]
NUM_CLASSES = len(CLASS_NAMES)

app = Flask(__name__)
model = None

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
            print(f"*** Modelo '{MODEL_PATH}' cargado exitosamente. ***")
            try:
                output_shape = model.output_shape
                if isinstance(output_shape, list):
                    num_model_outputs = output_shape[0][-1]
                    print(f"Modelo detectado con múltiples salidas. Verificando la primera: {num_model_outputs} neuronas.")
                else:
                     num_model_outputs = output_shape[-1]
                if num_model_outputs != NUM_CLASSES:
                     print(f"¡ADVERTENCIA GRAVE!")
                     print(f"El modelo cargado tiene {num_model_outputs} neuronas en su capa de salida.")
                     print(f"Sin embargo, se definieron {NUM_CLASSES} clases en CLASS_NAMES.")
                else:
                     print(f"El modelo tiene {num_model_outputs} salidas, coincidiendo con las {NUM_CLASSES} clases definidas. ¡Correcto!")
            except Exception as e:
                print(f"Advertencia: No se pudo verificar la capa de salida del modelo automáticamente: {e}")
        except Exception as e:
            print(f"*** ERROR CRÍTICO al cargar el modelo '{MODEL_PATH}': {e} ***")
            model = None
    else:
        print(f"*** ERROR CRÍTICO: No se encontró el archivo del modelo en '{MODEL_PATH}'. La API no podrá realizar predicciones. ***")

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('L')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype('float32') / 255.0
        if img_array.shape[-1] != 1:
            img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"*** ERROR durante el preprocesamiento de la imagen: {e} ***")
        return None

@app.route('/')
def index():
    load_trained_model()
    status = "Modelo cargado correctamente." if model is not None else "ERROR: Modelo NO cargado. Verifica los logs del servidor."
    return jsonify({
        "message": "API de Detección de Fatiga Visual (Clasificación de Imágenes)",
        "model_status": status,
        "expected_image_input": {
             "width": IMAGE_WIDTH,
             "height": IMAGE_HEIGHT,
             "channels": IMAGE_CHANNELS,
             "color_mode": COLOR_MODE
        },
        "available_classes": CLASS_NAMES
    })

@app.route('/predict', methods=['POST'])
def predict():
    load_trained_model()
    if model is None:
        return jsonify({"error": "El modelo de predicción no está disponible en este momento. Revisa los logs del servidor."}), 503
    if 'file' not in request.files:
        return jsonify({"error": "Petición inválida: No se encontró la parte del archivo ('file') en la solicitud."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Petición inválida: No se seleccionó ningún archivo."}), 400
    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Archivo recibido está vacío."}), 400
    except Exception as e:
        return jsonify({"error": f"No se pudo leer el archivo de imagen: {e}"}), 400
    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        return jsonify({"error": "Ocurrió un error durante el preprocesamiento de la imagen. Verifica los logs."}), 400
    try:
        predictions_raw = model.predict(processed_image)
        probabilities = predictions_raw[0]
        predicted_index = np.argmax(probabilities)
        confidence = float(probabilities[predicted_index])
        if 0 <= predicted_index < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_index]
        else:
            predicted_class = f"Error: Índice fuera de rango ({predicted_index})"
            confidence = 0.0
        result = {
        "success": True,
        "label": predicted_class,
        "score": round(confidence, 5),
    
          }
        return jsonify(result), 200
    except Exception as e:
        print(f"*** ERROR CRÍTICO durante la predicción o procesamiento del resultado: {e} ***")
        return jsonify({"error": f"Error interno del servidor durante la predicción: {e}"}), 500

if __name__ == '__main__':
    print("+" + "-"*60 + "+")
    print("| Iniciando API de Clasificación de Imágenes de Fatiga Visual |")
    print("+" + "-"*60 + "+")
    print("Cargando modelo de TensorFlow/Keras...")
    load_trained_model()
    print("-" * 62)
    if model is not None:
        print("Modelo listo. Iniciando servidor Flask...")
        app.run(host='0.0.0.0', port=10000, debug=True)
    else:
        print("ERROR: El modelo no se pudo cargar. El servidor Flask NO se iniciará.")
