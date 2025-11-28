Traffic volumes prediction using LSTM model.\n
-Used to predict future traffic on 22 features.
-Can indicate rushes forming.
-Helps plan best time or routes for journey.
-Can be used to plan signal timings.

It learns from:
-hour
-day of week
-temperature
-rain/snow
-weather
-past traffic (lags + rolling averages)

It can understand real-world patterns like:
-morning rush hour
-evening peak
-weekend low traffic
-weather-related slowdowns

Future Development:
1.Adding Actual Traffic from IoT sensors, alarming system, more accurate data for prediction.
Integration of-
-IoT sensors (cameras, loops, radar)
-Edge gateway (process & forward data)
-Network layer (Wi-Fi, 4G/5G, or LoRa)
-Backend server (Flask API / database)
-Preprocessing pipeline (convert raw data → model features)
-Model integration (predict traffic using LSTM)
-Dashboard / visualization (show live predictions)
