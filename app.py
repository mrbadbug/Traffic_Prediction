from flask import Flask, render_template, jsonify
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model and scalers
model = load_model("models/traffic_lstm_model.h5", compile=False)
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# Simulation parameters
SEQ_LENGTH = 10
FEATURES = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description']
roads = ['Road_A', 'Road_B', 'Road_C']

# Store simulated data
traffic_data = {road: {'timestamps': [], 'actual': [], 'predicted': []} for road in roads}

# Simulated IoT sequence memory per road
last_sequences = {
    road: np.zeros((SEQ_LENGTH, len(FEATURES))) for road in roads
}

def simulate_traffic():
    """Simulate IoT updates and model predictions every 5 seconds."""
    while True:
        current_time = time.strftime("%H:%M:%S")

        for road in roads:
            # Simulate IoT sensor readings (temperature, weather, etc.)
            new_data = np.array([[0, 25 + np.random.randn(), 0, 0, 40 + np.random.randn(), 1, 3]])
            new_data_scaled = scaler_X.transform(new_data)

            # Update the sequence buffer
            seq = last_sequences[road]
            seq = np.vstack([seq[1:], new_data_scaled])
            last_sequences[road] = seq

            # Predict next traffic volume
            pred_scaled = model.predict(seq.reshape(1, SEQ_LENGTH, len(FEATURES)), verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]

            # Simulate an actual traffic measurement with small random noise
            actual = pred + np.random.normal(0, 30)

            # Store results
            traffic_data[road]['timestamps'].append(current_time)
            traffic_data[road]['predicted'].append(pred)
            traffic_data[road]['actual'].append(actual)

            # Keep only the last 50 data points
            if len(traffic_data[road]['timestamps']) > 50:
                for key in ['timestamps', 'predicted', 'actual']:
                    traffic_data[road][key] = traffic_data[road][key][-50:]

        time.sleep(5)

# Start background thread
threading.Thread(target=simulate_traffic, daemon=True).start()

@app.route("/")
def dashboard():
    return render_template("dashboard.html", roads=roads)

@app.route("/data")
def data():
    response = {'timestamps': traffic_data[roads[0]]['timestamps']}
    for road in roads:
        response[road + '_actual'] = [float(x) for x in traffic_data[road]['actual']]
        response[road + '_predicted'] = [float(x) for x in traffic_data[road]['predicted']]
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

