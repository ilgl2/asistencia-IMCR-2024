import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import time
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import SpatialDropout2D
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import os
import pyttsx3

def tarea_2E(epoch , batch, verbose):

#     Cargar el modelo pre-entrenado
    model = Xception(weights='imagenet')

#     Inicializar el motor de síntesis de voz
    engine = pyttsx3.init()

#     Cargar una imagen de ejemplo
    img_path = 'imagen8.jpg'
    img = image.load_img(img_path, target_size=(299, 299))

#     Convertir la imagen a un arreglo numpy
    x = image.img_to_array(img)

#     Agregar una dimensión extra para que coincida con el formato de entrada del modelo
    x = np.expand_dims(x, axis=0)

#     Preprocesar la imagen para que coincida con el preprocesamiento que se hizo durante el entrenamiento
    x = preprocess_input(x)

#     Realizar predicciones
    preds = model.predict(x)

#     Decodificar las predicciones en etiquetas humanas
    predictions = decode_predictions(preds, top=3)[0]

#     Imprimir las predicciones
    print('Predicción:', predictions)

#     Convertir las predicciones a texto
#     text_to_speak = 'Esta viendo '
    countabove_0_2 = sum(1 for (_, _, prob) in predictions if prob > 0.2)
    aux = 0
    text_to_speak = 'You are looking a '
    for (imagenet_id, label, prob) in predictions:
#         text_to_speak += f'{label}: {prob:.2f}, '
        if (prob > 0.1):
            if(aux <= countabove_0_2 and aux != 0):
                text_to_speak += f' or '
            text_to_speak += f'{label}'
        aux = aux + 1
#     Reproducir el texto por audio
    engine.say(text_to_speak)
    engine.runAndWait()
        
def main():
    tarea_2E(epoch = 15, batch = 64, verbose = True)
    return 0

if __name__ == "__main__":
    main()