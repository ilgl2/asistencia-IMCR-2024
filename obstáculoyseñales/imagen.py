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
import os

def tarea_2E(epoch , batch, verbose):
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras
    (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
    
    # Guardar los datos en archivos Numpy
    #np.save('train_images_100.npy', X_train)
    #np.save('train_labels_100.npy', Y_train)
    #np.save('test_images_100.npy', X_test)
    #np.save('test_labels_100.npy', Y_test)
    
    # Cargar los datos desde archivos Numpy
    #X_train = np.load('train_images_100.npy')
    #Y_train = np.load('train_labels_100.npy')
    #X_test = np.load('test_images_100.npy')
    #Y_test = np.load('test_labels_100.npy')

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    Y_train = to_categorical(Y_train, 10)
    Y_test = to_categorical(Y_test, 10)

    #Guardamos el tiempo de inicio
    startTime = time.time()

     #Creamos el modelo
    model = Sequential()
    
    #Añadimos una capa de Conv con 32 neuronas y escribimos las dimensiones de las imágenes de entrada
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    
    #Añadimos una capa de MaxPooling2D
    model.add(MaxPooling2D((2, 2)))
    
    #Añadimos otra capa de Conv2D de 64 neuronas con activación relu
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(SpatialDropout2D(0.2))

    #Aplanamos los datos
    model.add(Flatten())

    #Capa con 64 neuronal totalmente conectada
    model.add(Dense(256, activation='relu'))
    
    model.add(Dropout(0.3))
    
    #Capa de salida con 10 neuronas
    model.add(Dense(10, activation='softmax'))

    optimizador = Adam(learning_rate=0.005)
    
    #Usamos compile para compilar el modelo
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizador, metrics=['accuracy'])
    
    # Compilar el modelo
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    # Verificar si existen pesos previamente entrenados
    if os.path.exists('modelo_entrenado_10.weights.h5'):
        print("Cargando pesos previamente entrenados...")
        model.load_weights('modelo_entrenado_10.weights.h5')
    else:
        history = model.fit(X_train, Y_train, epochs=epoch, validation_data=(X_test, Y_test), validation_split=0.1)
        # Guardar los pesos del modelo
        model.save_weights('modelo_entrenado_10.weights.h5')
        print("Pesos del modelo guardados.")
    
    #Entrenamos el modelo
    #model.fit(X_train, Y_train, epochs=epoch, batch_size=batch, validation_split=0.1)

    #Una vez entrenado usamos predict
    Y_pred = model.predict(X_test)
    
    # Convertimos las etiquetas one-hot a etiquetas de clase originales
    Y_test_labels = np.argmax(Y_test, axis=1)
    
    #Elegimos los que tienen valor más alto
    Y_pred_final = np.argmax(Y_pred, axis=1)
    
    #Calculamos la precisión
    precisionTest = np.mean(Y_pred_final == Y_test_labels)
    
    if verbose == True:
        #Imprimimos la matriz de confusión
        print(confusion_matrix(Y_test_labels, Y_pred_final))
        
    #Guardamos el tiempo fin
    endTime = time.time()

    #Evaluamos el modelo
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    imagen = cv2.imread("imagen.jpg")
    imagen = cv2.resize(imagen, (32, 32))  # Ajusta el tamaño según tu modelo
    imagen = np.reshape(imagen, [1, 32, 32, 3])  # 3 canales (RGB)

    # Obtener las probabilidades de las clases
    probabilidades = model.predict(imagen)

    # Seleccionar la clase con la probabilidad más alta para cada imagen
    clases_predichas = [np.argmax(p) for p in probabilidades]

    print(f"Clases predichas: {clases_predichas}")

    #Sacamos el tiempo total restando el tiempo fin menos el tiempo de inicio
    totalTime = endTime - startTime
    print(f"Precisión en el conjunto de prueba: {accuracy}")
    print(f"Tasas acierto test y tiempo: {precisionTest:.2%}, {totalTime:.3f} s con loss {loss:.4f} y accuracy {accuracy:.4f}.")

    
def main():
    tarea_2E(epoch = 15, batch = 64, verbose = True)
    return 0

if __name__ == "__main__":
    main()