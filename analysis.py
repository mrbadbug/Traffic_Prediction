import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

# Encode categorical features
df = pd.get_dummies(df, columns=['weather_main', 'weather_description'], drop_first=True)

# Features and target
X = df.drop(['traffic_volume', 'date_time'], axis=1)
y = df['traffic_volume']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R2 score:", r2_score(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
indices = importances.argsort()[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()
