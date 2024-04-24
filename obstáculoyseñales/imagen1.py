import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import time
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras import datasets
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def tarea_2E(epoch , batch):
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de Yann Lecun)
    (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
    
    # Guardar los datos en archivos Numpy
    #np.save('train_images.npy', X_train)
    #np.save('train_labels.npy', Y_train)
    #np.save('test_images.npy', X_test)
    #np.save('test_labels.npy', Y_test)
    
    # Cargar los datos desde archivos Numpy
    #X_train = np.load('train_images.npy')
    #Y_train = np.load('train_labels.npy')
    #X_test = np.load('test_images.npy')
    #Y_test = np.load('test_labels.npy')
    
    # Normalizar las imágenes
    X_train, X_test = X_train / 255.0, X_test / 255.0

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
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    #Aplanamos los datos
    model.add(Flatten())
    
    #Capa con 64 neuronal totalmente conectada
    model.add(Dense(64, activation='relu'))
    
    #Capa de salida con 10 neuronas
    model.add(Dense(10, activation='softmax'))

    optimizador = Adam(learning_rate=0.001)
    
    #Usamos compile para compilar el modelo
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizador, metrics=['accuracy'])
    
    history = model.fit(X_train, Y_train, epochs=epoch, validation_data=(X_test, Y_test))
    
    #Entrenamos el modelo
    #model.fit(X_train, Y_train, epochs=epoch, batch_size=batch, validation_split=0.1)

    #Una vez entrenado usamos predict
    Y_pred = model.predict(X_test)
    
    #Elegimos los que tienen valor más alto
    Y_pred_final = np.argmax(Y_pred, axis=1)
    
    #Calculamos la precisión
    precisionTest = np.mean(Y_pred_final == Y_test)
    
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
    tarea_2E(epoch = 10, batch = 32)
    return 0

if __name__ == "__main__":
    main()
