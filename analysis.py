import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.activations import softplus



# Load dataset
df = pd.read_csv("/content/drive/MyDrive/Metro_Interstate_Traffic_Volume.csv")
df['date_time'] = pd.to_datetime(df['date_time'])
df.set_index('date_time', inplace=True)

# Encode categorical features
le_holiday = LabelEncoder()
df['holiday'] = le_holiday.fit_transform(df['holiday'])

le_weather = LabelEncoder()
df['weather_main'] = le_weather.fit_transform(df['weather_main'])
df['weather_description'] = le_weather.fit_transform(df['weather_description'])

# Fill missing values
df.fillna(0, inplace=True)

# Features and target
features = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description']
target = ['traffic_volume']

X = df[features].values
y = df[target].values

# Normalize
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Create sequences
SEQ_LENGTH = 10
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X)-seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)

# Split train/test
train_size = int(len(X_seq)*0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Build LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(SEQ_LENGTH, X_seq.shape[2])))
model.add(Dense(1, activation=softplus))
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)
model.save("models/traffic_lstm_model.h5")

# Save scalers
import joblib
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# Predict and inverse transform
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Ensure no negative predictions
y_pred = np.maximum(0, y_pred)
