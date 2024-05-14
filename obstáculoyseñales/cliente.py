import requests
import json

def enviar_imagen():
    url_servidor = ""  # Servidor url
    archivo_imagen = "./imagen.jpg"  # Ruta de la imagen

    # Cargamos la imagen
    with open(archivo_imagen, "rb") as archivo:
        imagen_base64 = archivo.read().encode("base64").decode()

    # Creamos el objeto JSON
    payload = {"imagen": imagen_base64}

    # Envía la solicitud al servidor
    try:
        respuesta = requests.post(url_servidor, json=payload)
        if respuesta.status_code == 200:
            print("Imagen enviada correctamente.")
        else:
            print(f"Error al intentar enviar la imagen: {respuesta.status_code}")
    except requests.RequestException as e:
        print(f"Error de conexión: {e}")

if __name__ == "__main__":
    enviar_imagen()
