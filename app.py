import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_prediction_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Define possible diabetes management states
DIABETES_STATES = [
    "Increase Insulin",
    "Decrease Insulin",
    "Maintain Dosage",
    "Lifestyle Change Required",
    "Urgent Doctor Visit"
]

app = Flask(__name__)

@app.route('/')
def home():
    return "Diabetes Prediction Model API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expect JSON input
        data = request.get_json()

        # Convert input to NumPy array (ensure correct shape)
        input_data = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)  # Get the class index

        # Get the corresponding state
        result = {
            "predicted_state": DIABETES_STATES[predicted_class],
            "confidence_scores": prediction.tolist()  # Send all class probabilities
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    # Add this at the end of the file
application = app  # For WSGI compatibility