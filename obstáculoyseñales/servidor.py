import base64
from flask import request

@app.route("/classify/", methods=["POST"])
def classify():
    data = request.get_json()
    if "imagen" in data:
        imagen_base64 = data["imagen"]
        imagen_bytes = base64.b64decode(imagen_base64)


        return jsonify({"mensaje": "Imagen recibida correctamente"}), 200