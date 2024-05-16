import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import pyttsx3

def redNeuronal():

    #Cargamos el modelo pre-entrenado
    model = Xception(weights='imagenet')

    #Inicializamos el motor de síntesis de voz
    engine = pyttsx3.init()

    #Cargamos una imagen de ejemplo
    img_path = 'imagen1.jpg'
    img = image.load_img(img_path, target_size=(299, 299))

    #Convertimos la imagen a un arreglo numpy
    x = image.img_to_array(img)

    #Agregamos una dimensión extra para que coincida con el formato de entrada del modelo
    x = np.expand_dims(x, axis=0)

    #Preprocesamos la imagen para que coincida con el preprocesamiento que se hizo durante el entrenamiento
    x = preprocess_input(x)

    #Realizamos predicciones
    preds = model.predict(x)

    #Decodificamos las predicciones en etiquetas humanas
    predictions = decode_predictions(preds, top=5)[0]

    #Imprimimos las predicciones
    print('Predicción:', predictions)

    #Convertimos las predicciones a texto
    countabove_0_2 = sum(1 for (_, _, prob) in predictions if prob > 0.2)
    aux = 0
    text_to_speak = 'You are looking a '
    for (imagenet_id, label, prob) in predictions:
        if (prob >= 0.02):
            if(aux <= countabove_0_2 and aux != 0):
                text_to_speak += ' and '
            text_to_speak += label.replace('_', ' ')  # Reemplaza todos los guiones bajos por espacios
        aux = aux + 1
    
    #Reproducimos el texto por audio
    engine.say(text_to_speak)
    engine.runAndWait()
        
def main():
    redNeuronal()
    return 0

if __name__ == "__main__":
    main()