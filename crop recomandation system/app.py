from flask import Flask, request, render_template
import joblib
import numpy as np

# model loading
model = joblib.load("crop_recommendation_model.pkl")
scaler= joblib.load("scaler.pkl")
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("input.html")

def validate_input(n, p, k, temperature, humidity, ph, rainfall):
    if not (0 <= n <= 140): raise ValueError("N must be between 0–140")
    if not (5 <= p <= 145): raise ValueError("P must be between 5–145")
    if not (5 <= k <= 205): raise ValueError("K must be between 5–205")
    if not (10 <= temperature <= 45): raise ValueError("Temperature must be 10–45°C")
    if not (20 <= humidity <= 100): raise ValueError("Humidity must be 20–100%")
    if not (3.5 <= ph <= 9.5): raise ValueError("pH must be 3.5–9.5")
    if not (20 <= rainfall <= 300): raise ValueError("Rainfall must be 20–300mm")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        N = request.form.get("N", type=float)
        P = request.form.get("P", type=float)
        K = request.form.get("K", type=float) 
        T = request.form.get("temperature", type=float)
        H = request.form.get("humidity", type=float)
        ph = request.form.get("ph", type=float)
        rain = request.form.get("rainfall", type=float)
        # Validate input
        validate_input(N, P, K, T, H, ph, rain)
        features = [N, P, K, T, H, ph, rain]

        input_array = np.array([features])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        return render_template("input.html", prediction=prediction)

    except Exception as e:
         return render_template("input.html", prediction=f"Error: {e}", values=request.form)

if __name__ == "__main__":
    app.run(debug=True)
