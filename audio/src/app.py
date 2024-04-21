import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import ai.classification as classification

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = "./audio/data"

@app.route("/classify/", methods=["POST"])
def classify():
    files = request.files
    response = []
    print(files)
    for filename in files:
        filename_secure = secure_filename(filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename_secure)
        files[filename].save(path)
        try:
            label = classification.classify(path)
            response.append({filename: label})
        except:
            print(f"Error processing file: {path}")
        os.remove(path)
    return jsonify(response), 200

app.run("0.0.0.0", 8000, debug=True, threaded=True)