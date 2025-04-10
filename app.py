# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)

# Cargar modelo
modelo = tf.keras.models.load_model("modelo_digitos.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediccion = modelo.predict(image)
    resultado = int(np.argmax(prediccion))

    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)
