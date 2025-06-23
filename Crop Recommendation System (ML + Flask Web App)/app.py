from flask import Flask, request, render_template
import joblib
import numpy as np

# model loading
model = joblib.load("crop_recommendation_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        features = [
           request.form.get("N", type=float),
           request.form.get("P", type=float),
           request.form.get("K", type=float),
           request.form.get("temperature", type=float),
           request.form.get("humidity", type=float),
           request.form.get("ph", type=float),
           request.form.get("rainfall", type=float)
        ]

        
        input_array = np.array([features])
        prediction = model.predict(input_array)[0]

        return render_template("input.html", prediction=prediction)

    except Exception as e:
        return render_template("input.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
