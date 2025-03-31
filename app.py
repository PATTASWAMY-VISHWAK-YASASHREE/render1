

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Define the custom model class and register it
@tf.keras.utils.register_keras_serializable()
class DiabetesModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(DiabetesModel, self).__init__(*args, **kwargs)

# Load the trained model with custom objects
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_prediction_model.keras")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"DiabetesModel": DiabetesModel})

# Define possible diabetes management states
DIABETES_STATES = [
    "Increase Insulin",
    "Decrease Insulin",
    "Maintain Dosage",
    "Lifestyle Change Required",
    "Urgent Doctor Visit"
]

# Explanations for each state
EXPLANATIONS = {
    "Increase Insulin": "Your insulin levels are too low, requiring an increase in dosage.",
    "Decrease Insulin": "Your insulin levels are too high, requiring a reduction in dosage.",
    "Maintain Dosage": "Your insulin and glucose levels are stable; continue with your current dosage.",
    "Lifestyle Change Required": "Your glucose levels indicate a need for diet and exercise adjustments.",
    "Urgent Doctor Visit": "Your glucose and insulin levels are critically abnormal. Seek medical help immediately."
}

app = Flask(__name__)

@app.route('/')
def home():
    return "Diabetes Prediction Model API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        predicted_class = np.argmax(prediction)  
        predicted_state = DIABETES_STATES[predicted_class]

        result = {
            "predicted_state": predicted_state,
            "confidence_scores": prediction.tolist(),
            "explanation": EXPLANATIONS[predicted_state]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

application = app
