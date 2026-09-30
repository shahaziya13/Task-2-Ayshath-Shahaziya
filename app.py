import streamlit as st
import pandas as pd
import joblib


# ======================================
# PAGE CONFIGURATION
# ======================================

st.set_page_config(
    page_title="FLOODGUARD AI",
    page_icon="🌊",
    layout="centered"
)


# ======================================
# LOAD MODEL
# ======================================

model = joblib.load("models/floodguard_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


# ======================================
# HEADER
# ======================================

st.title("🌊 FLOODGUARD AI")
st.subheader("AI-Based Flood Risk Classification System")

st.write(
    "Enter environmental conditions below to classify "
    "the flood risk as Low, Moderate, or High."
)

st.info(
    "⚠️ Educational prototype: This system uses a synthetic dataset "
    "and is not intended for real-world emergency or flood warnings."
)


# ======================================
# INPUT SECTION
# ======================================

st.header("🌧️ Environmental Conditions")

col1, col2 = st.columns(2)

with col1:

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=1.0
    )

    river_level = st.number_input(
        "River Level (m)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.1
    )

    soil_moisture = st.number_input(
        "Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=25.0,
        step=0.5
    )


with col2:

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=200.0,
        value=20.0,
        step=1.0
    )

    previous_rainfall = st.number_input(
        "Previous Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    elevation = st.number_input(
        "Elevation (m)",
        min_value=0.0,
        max_value=5000.0,
        value=100.0,
        step=1.0
    )

    drainage_capacity = st.number_input(
        "Drainage Capacity (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )


# ======================================
# PREDICTION BUTTON
# ======================================

if st.button("🔍 Predict Flood Risk", use_container_width=True):

    # Create input dataframe
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


    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    predicted_class = label_encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = max(probabilities) * 100


    # ======================================
    # PREDICTION RESULT
    # ======================================

    st.header("📊 Prediction Result")

    if predicted_class == "High":

        st.error("🔴 HIGH FLOOD RISK")

    elif predicted_class == "Moderate":

        st.warning("🟠 MODERATE FLOOD RISK")

    else:

        st.success("🟢 LOW FLOOD RISK")


    st.write(
        f"**Model Confidence: {confidence:.2f}%**"
    )


    # ======================================
    # INPUT SUMMARY
    # ======================================

    st.subheader("📋 Input Summary")

    summary_df = pd.DataFrame({
        "Parameter": [
            "Rainfall",
            "River Level",
            "Soil Moisture",
            "Humidity",
            "Temperature",
            "Wind Speed",
            "Previous Rainfall",
            "Elevation",
            "Drainage Capacity"
        ],

        "Value": [
            f"{rainfall} mm",
            f"{river_level} m",
            f"{soil_moisture} %",
            f"{humidity} %",
            f"{temperature} °C",
            f"{wind_speed} km/h",
            f"{previous_rainfall} mm",
            f"{elevation} m",
            f"{drainage_capacity} %"
        ]
    })

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True
    )


    # ======================================
    # CLASSIFICATION PROBABILITIES
    # ======================================

    st.subheader("📈 Classification Probabilities")

    probability_df = pd.DataFrame({
        "Risk Category": label_encoder.classes_,
        "Probability": probabilities * 100
    })

    # Clean probability display
    for category, probability in zip(
        label_encoder.classes_,
        probabilities
    ):

        percentage = float(probability * 100)

        st.write(
            f"**{category} Risk — {percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )


    # ======================================
    # FEATURE IMPORTANCE
    # ======================================

    st.subheader("🔎 Important Factors")

    feature_importance = pd.DataFrame({
        "Factor": model.feature_names_in_,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    for _, row in feature_importance.iterrows():

        factor_name = (
            row["Factor"]
            .replace("_", " ")
            .title()
        )

        importance = float(row["Importance"])

        st.write(
            f"**{factor_name} — {importance * 100:.2f}%**"
        )

        st.progress(
            importance
        )


    # ======================================
    # MODEL INFORMATION
    # ======================================

    st.subheader("🤖 Model Information")

    model_info = pd.DataFrame({
        "Property": [
            "Algorithm",
            "Learning Type",
            "Number of Features",
            "Output Classes"
        ],

        "Details": [
            "Random Forest Classifier",
            "Supervised Learning",
            "9",
            "Low, Moderate, High"
        ]
    })

    st.dataframe(
        model_info,
        hide_index=True,
        use_container_width=True
    )


# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.caption(
    "FLOODGUARD AI | Data Classification Using AI | "
    "Educational Project"
)