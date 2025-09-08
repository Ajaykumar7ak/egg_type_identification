import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from egg_info import egg_info

app = Flask(__name__)

# Load trained model
MODEL_PATH = "models/egg_model.keras"
model = load_model(MODEL_PATH)

# Class labels (must match your training order)
class_names = ["chicken", "duck", "emu", "ostrich", "turkey"]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict(filepath):
    img = image.load_img(filepath, target_size=(224, 224))  # same size as training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    return class_names[np.argmax(preds)]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    details = None
    uploaded_file = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            uploaded_file = filepath
            prediction = predict(filepath)
            details = egg_info.get(prediction, {})

    return render_template("index.html",
                           prediction=prediction,
                           details=details,
                           uploaded_file=uploaded_file)

if __name__ == "__main__":
    app.run(debug=True)
