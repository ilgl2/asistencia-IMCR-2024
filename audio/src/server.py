from flask import Flask, request, jsonify

import ai.classification as classification

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = "./audio/data"

@app.route("/classify/", methods=["POST"])
def classify():
    files = request.files
    response = []
    for filename in files:
        try:
            label = classification.classify(files[filename])
            response.append({filename: label})
        except:
            print(f"Error processing file: {filename}")
    return jsonify(response), 200

app.run("0.0.0.0", 8000, debug=True, threaded=True)