from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import threading
import time

app = Flask(__name__)

# Load model and scalers
model = load_model("models/traffic_lstm_model.keras")
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

SEQ_LENGTH = 10
FEATURES = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description']

# Load dataset
df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")
df['date_time'] = pd.to_datetime(df['date_time'])
df.set_index('date_time', inplace=True)
df.fillna(0, inplace=True)

# Simulate multiple roads
roads = ['Road_A', 'Road_B', 'Road_C']
road_sequences = {}
road_predictions = {}

for road in roads:
    last_seq = df[FEATURES].values[-SEQ_LENGTH:]
    road_sequences[road] = scaler_X.transform(last_seq)
    road_predictions[road] = []

# Function to simulate IoT data for each road
def simulate_iot_data(road):
    holiday = 0
    temp = 20 + np.random.randn()*2
    rain_1h = max(0, np.random.randn())
    snow_1h = max(0, np.random.randn()*0.5)
    clouds_all = min(100, max(0, 50 + np.random.randn()*10))
    weather_main = 1
    weather_description = 3
    return np.array([[holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description]])

# Real-time prediction thread
def realtime_prediction():
    while True:
        for road in roads:
            new_data = simulate_iot_data(road)
            new_data_scaled = scaler_X.transform(new_data)
            seq = road_sequences[road]
            seq = np.vstack([seq[1:], new_data_scaled])
            pred_scaled = model.predict(seq.reshape(1, SEQ_LENGTH, len(FEATURES)))
            pred_traffic = scaler_y.inverse_transform(pred_scaled)
            road_sequences[road] = seq
            road_predictions[road].append(pred_traffic[0][0])
            if len(road_predictions[road]) > 50:
                road_predictions[road] = road_predictions[road][-50:]
        time.sleep(5)

threading.Thread(target=realtime_prediction, daemon=True).start()

@app.route("/")
def home():
    return render_template("dashboard.html", roads=roads)

@app.route("/data")
def data():
    timestamps = df.index[-50:].strftime("%Y-%m-%d %H:%M:%S").tolist()
    data_dict = {"timestamps": timestamps}
    for road in roads:
        actual = df['traffic_volume'].tail(50).tolist()
        predicted = road_predictions[road]
        data_dict[f"{road}_actual"] = actual
        data_dict[f"{road}_predicted"] = predicted
    return jsonify(data_dict)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
