import requests
import pyttsx3

# Envía la solicitud al servidor
output = None
try:
    image = open("./obstacles/data/imagen.jpg", "rb")
    respuesta = requests.post("http://localhost:8001/classify/image/", files={"imagen.jpg": image})
    image.close()
    if respuesta.status_code == 200:
        print("Imagen procesada correctamente.")
        output = respuesta.json()[0]["text"]
    else:
        print(f"Error al procesar la imagen: {respuesta.json()}")
        output = ""
except requests.RequestException as e:
    print(f"Error de conexión: {e}")

#Inicializamos el motor de síntesis de voz
engine = pyttsx3.init()
engine.say(output)
engine.runAndWait()