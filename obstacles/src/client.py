import requests
import pyttsx3

# Envía la solicitud al servidor
obstacles = None
transcription = None

try:
    image = open("./obstacles/data/imagen6.jpg", "rb")
    respuesta = requests.post("http://localhost:8001/classify/image/", files={"imagen.jpg": image})
    image.close()

    engine = pyttsx3.init()
    if respuesta.status_code == 200:
        print("Imagen procesada correctamente.")
        response = respuesta.json()
        print(response)
        if len(response) > 0:
            obstacles = response[0]["text"]
            if obstacles:
                engine.say(obstacles)
            transcription = response[0]["transcription"]
            if transcription:
                engine.say(f"There is a sign with the following text: {transcription}")

            engine.runAndWait()
    else:
        print(f"Error al procesar la imagen: {respuesta.json()}")
except requests.RequestException as e:
    print(f"Error de conexión: {e}")