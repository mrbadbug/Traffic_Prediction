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
FEATURES = [
    'holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
    'weather_main', 'weather_description',
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'traffic_lag1', 'traffic_lag2',
    'traffic_rolling3', 'rain_3h', 'snow_3h', 'temp_3h'
]
roads = ['Road_A', 'Road_B', 'Road_C']

# Store simulated data
traffic_data = {road: {'timestamps': [], 'predicted': []} for road in roads}

# sequence memory per road
last_sequences = {
    road: np.zeros((SEQ_LENGTH, len(FEATURES))) for road in roads
}

def simulate_traffic():

    while True:
        current_time = time.strftime("%H:%M:%S")

        for road in roads:
            # Generate random data for simulation
            new_data_scaled = np.random.rand(1, len(FEATURES))

            # Update the sequence buffer
            seq = last_sequences[road]
            seq = np.vstack([seq[1:], new_data_scaled])
            last_sequences[road] = seq

            # Predict next traffic volume
            pred_scaled = model.predict(seq.reshape(1, SEQ_LENGTH, len(FEATURES)), verbose=0)

            # Inverse transform safely
            pred = float(scaler_y.inverse_transform(pred_scaled)[0][0])
            pred = max(pred, 0)  # <--- STOP NEGATIVES

            # Store results
            traffic_data[road]['timestamps'].append(current_time)
            traffic_data[road]['predicted'].append(pred)

            # Keep only last 50
            if len(traffic_data[road]['timestamps']) > 50:
                for key in ['timestamps', 'predicted']:
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
        # Convert np.float32 to float for JSON serialization
        response[road + '_predicted'] = [float(x) for x in traffic_data[road]['predicted']]
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


