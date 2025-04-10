import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math

# Cargar datos MNIST
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)
datos_entrenamiento = datos['train']
datos_prueba = datos['test']

# Normalizar
def normalizar(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta

datos_entrenamiento = datos_entrenamiento.map(normalizar).cache()
datos_prueba = datos_prueba.map(normalizar).cache()

# Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),       # 784 nodos (28x28)
    tf.keras.layers.Dense(128, activation='relu'),          # 128 nodos en la capa oculta
    tf.keras.layers.Dense(10, activation='softmax')         # 10 nodos de salida (0 al 9)
])

# Hiperparámetros
epochs = 10
batch_size = 32
learning_rate = 0.005
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Preparar batches
num_img_entrenamiento = metadatos.splits['train'].num_examples
num_img_prueba = metadatos.splits['test'].num_examples
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_img_entrenamiento).batch(batch_size)
datos_prueba = datos_prueba.batch(batch_size)

# Entrenamiento
steps_per_epoch = math.ceil(num_img_entrenamiento / batch_size)
modelo.fit(datos_entrenamiento, epochs=epochs, steps_per_epoch=steps_per_epoch)

# Evaluar
test_loss, test_accuracy = modelo.evaluate(datos_prueba, steps=math.ceil(num_img_prueba / batch_size))
print(f"Precisión en datos de prueba: {test_accuracy*100:.2f}%")

# Guardar modelo
modelo.save("modelo_digitos.h5")
