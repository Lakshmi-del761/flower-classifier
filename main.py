from flask import Flask, render_template, request, jsonify
import os
from predict import predict_flower

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------
# Home Page
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# Web Prediction
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    flower, confidence = predict_flower(filepath)

    return render_template(
        "index.html",
        prediction=flower,
        confidence=round(confidence * 100, 2),
        image=file.filename
    )


# -------------------------
# REST API Prediction
# -------------------------
@app.route("/api/predict", methods=["POST"])
def predict_api():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    flower, confidence = predict_flower(filepath)

    return jsonify({
        "prediction": flower,
        "confidence": round(confidence * 100, 2)
    })


# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)


























