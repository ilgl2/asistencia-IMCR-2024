from flask import Flask, request, jsonify

import ai.reader as reader
import ai.classification as classification

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = "./obstacles/data"

@app.route("/classify/image/", methods=["POST"])
def classify():
    files = request.files
    response = []
    for filename in files:
        # try:
            preditiction = classification.classify(files[filename])
            transcription = reader.read(files[filename])
            response.append({"filename": filename, "text": preditiction, "transcription": transcription})
        # except:
            print(f"Error processing file: {filename}")
    return jsonify(response), 200

app.run("0.0.0.0", 8001, debug=True, threaded=True)