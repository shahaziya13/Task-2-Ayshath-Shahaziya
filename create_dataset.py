import pandas as pd
import numpy as np

np.random.seed(42)

n = 600

rainfall = np.random.uniform(5, 300, n)
river_level = np.random.uniform(0.5, 8.0, n)
soil_moisture = np.random.uniform(15, 95, n)
humidity = np.random.uniform(35, 100, n)
temperature = np.random.uniform(18, 40, n)
wind_speed = np.random.uniform(2, 45, n)
previous_rainfall = np.random.uniform(0, 250, n)
elevation = np.random.uniform(2, 100, n)
drainage_capacity = np.random.uniform(10, 100, n)

risk_score = (
    rainfall * 0.30
    + river_level * 8.0
    + soil_moisture * 0.18
    + humidity * 0.08
    + previous_rainfall * 0.16
    - elevation * 0.18
    - drainage_capacity * 0.12
    + wind_speed * 0.03
)

low_cut = np.percentile(risk_score, 40)
high_cut = np.percentile(risk_score, 75)

risk = np.where(
    risk_score < low_cut,
    "Low",
    np.where(risk_score < high_cut, "Moderate", "High")
)

df = pd.DataFrame({
    "rainfall_mm": np.round(rainfall, 2),
    "river_level_m": np.round(river_level, 2),
    "soil_moisture_percent": np.round(soil_moisture, 2),
    "humidity_percent": np.round(humidity, 2),
    "temperature_c": np.round(temperature, 2),
    "wind_speed_kmh": np.round(wind_speed, 2),
    "previous_rainfall_mm": np.round(previous_rainfall, 2),
    "elevation_m": np.round(elevation, 2),
    "drainage_capacity_percent": np.round(drainage_capacity, 2),
    "flood_risk": risk
})

df.to_csv("data/flood_risk_dataset.csv", index=False)

print("Dataset created successfully!")
print("Records:", len(df))
print(df["flood_risk"].value_counts())
