import pandas as pd
import joblib


# Load trained model and encoder
model = joblib.load("models/floodguard_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


print("\n======================================")
print("       FLOODGUARD AI PREDICTOR")
print("======================================")
print("Enter the environmental conditions.\n")


rainfall = float(input("Rainfall (mm): "))
river_level = float(input("River Level (m): "))
soil_moisture = float(input("Soil Moisture (%): "))
humidity = float(input("Humidity (%): "))
temperature = float(input("Temperature in Celsius: "))
wind_speed = float(input("Wind Speed (km/h): "))
previous_rainfall = float(input("Previous Rainfall (mm): "))
elevation = float(input("Elevation (m): "))
drainage_capacity = float(input("Drainage Capacity (%): "))


input_data = pd.DataFrame({
    "rainfall_mm": [rainfall],
    "river_level_m": [river_level],
    "soil_moisture_percent": [soil_moisture],
    "humidity_percent": [humidity],
    "temperature_c": [temperature],
    "wind_speed_kmh": [wind_speed],
    "previous_rainfall_mm": [previous_rainfall],
    "elevation_m": [elevation],
    "drainage_capacity_percent": [drainage_capacity]
})


prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

predicted_class = label_encoder.inverse_transform(
    [prediction]
)[0]

confidence = max(probabilities) * 100


print("\n======================================")
print("          PREDICTION RESULT")
print("======================================")

print(f"Flood Risk: {predicted_class.upper()}")
print(f"Model Confidence: {confidence:.2f}%")

print("======================================\n")
