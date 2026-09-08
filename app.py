import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Energy & Carbon Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f8fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 27px;
        font-weight: 650;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Cards */
    .metric-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #111827;
    }

    .metric-description {
        font-size: 13px;
        color: #6b7280;
        margin-top: 5px;
    }

    /* Information box */
    .info-box {
        background-color: white;
        border-left: 4px solid #374151;
        border-radius: 8px;
        padding: 18px;
        margin: 15px 0;
    }

    /* Result box */
    .result-box {
        background-color: #111827;
        color: white;
        padding: 28px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 15px;
        color: #d1d5db;
    }

    .result-value {
        font-size: 38px;
        font-weight: 700;
        color: white;
    }

    .result-unit {
        font-size: 16px;
        color: #d1d5db;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding: 30px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("ai_energy_model.pkl")


model = load_model()


# ============================================================
# COUNTRY CARBON INTENSITIES
# 2024 values from the project's energy dataset
# ============================================================

carbon_intensity = {
    "Norway": 30.75,
    "France": 40.83,
    "Brazil": 106.06,
    "Canada": 185.35,
    "United Kingdom": 216.50,
    "Germany": 338.36,
    "United States": 383.78,
    "Australia": 553.83,
    "China": 555.40,
    "India": 707.45
}


# ============================================================
# MODEL PERFORMANCE
# ============================================================

model_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
        "Deep Learning MLP"
    ],
    "MAE": [
        0.006567,
        0.006564,
        0.002143,
        0.002004,
        0.005988
    ],
    "RMSE": [
        0.009653,
        0.009651,
        0.002600,
        0.002342,
        0.008638
    ],
    "R²": [
        0.1429,
        0.1434,
        0.8749,
        0.9181,
        0.1601
    ]
})


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="font-size:24px;font-weight:700;margin-bottom:5px;">
        AI Energy Intelligence
    </div>
    <div style="font-size:13px;color:#9ca3af;margin-bottom:25px;">
        Predictive modelling and carbon analysis
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Energy Prediction",
        "Carbon Analysis",
        "Country Comparison",
        "Model Performance",
        "Feature Importance",
        "Methodology",
        "Limitations"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Research Model**

    Algorithm: Gradient Boosting Regressor

    Target: Energy Consumption

    Unit: kWh

    Validation: 5-fold GroupKFold
    """
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">AI Energy & Carbon Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Predicting the electricity consumption of AI inference workloads and estimating their associated carbon footprint.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Research Objective</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Artificial intelligence workloads require electricity to execute inference
        tasks. The amount of electricity consumed can vary according to workload
        characteristics such as request rate and output length.

        This application uses supervised machine learning to predict the energy
        consumption of an AI inference workload. The predicted energy consumption
        is subsequently combined with electricity-grid carbon intensity to estimate
        the associated carbon footprint.
        """
    )

    st.markdown(
        '<div class="info-box">'
        '<strong>Core prediction task:</strong><br>'
        'Predict the electricity consumption of an AI inference workload in kWh '
        'using request rate and output length.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Key Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Best Model</div>
                <div class="metric-value">GBR</div>
                <div class="metric-description">Gradient Boosting</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">0.918</div>
                <div class="metric-description">Mean 5-fold validation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">0.00234</div>
                <div class="metric-description">kWh</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Training Records</div>
                <div class="metric-value">1,024</div>
                <div class="metric-description">Completed workloads</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">How the Application Works</div>',
                unsafe_allow_html=True)

    workflow = pd.DataFrame({
        "Stage": [
            "Workload Input",
            "Energy Prediction",
            "Grid Carbon Intensity",
            "Carbon Estimation"
        ],
        "Description": [
            "Request rate and output length",
            "Gradient Boosting model predicts kWh",
            "Country-specific electricity carbon intensity",
            "Energy × carbon intensity"
        ]
    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ENERGY PREDICTION
# ============================================================

elif page == "Energy Prediction":

    st.markdown(
        '<div class="main-title">Energy Consumption Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Estimate the electricity required by an AI inference workload.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1.4])

    with left:

        st.markdown(
            '<div class="section-title">Workload Parameters</div>',
            unsafe_allow_html=True
        )

        request_rate = st.slider(
            "Request rate (requests/second)",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

        output_length = st.selectbox(
            "Output length (tokens)",
            [256, 512, 1024]
        )

        predict = st.button(
            "Predict Energy Consumption",
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="section-title">Prediction</div>',
            unsafe_allow_html=True
        )

        if predict:

            input_data = pd.DataFrame({
                "request_rate_x": [request_rate],
                "hf-output-len": [output_length]
            })

            prediction = float(model.predict(input_data)[0])

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">
                        Estimated Electricity Consumption
                    </div>
                    <div class="result-value">
                        {prediction:.6f}
                    </div>
                    <div class="result-unit">
                        kilowatt-hours (kWh)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            m1, m2 = st.columns(2)

            with m1:
                st.metric(
                    "Request Rate",
                    f"{request_rate:,} req/s"
                )

            with m2:
                st.metric(
                    "Output Length",
                    f"{output_length:,} tokens"
                )

            st.info(
                "The prediction is generated by the trained Gradient Boosting "
                "regression model using request rate and output length."
            )

        else:

            st.markdown(
                """
                <div class="info-box">
                    Enter the workload parameters and select
                    <strong>Predict Energy Consumption</strong> to generate
                    an estimate.
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# CARBON ANALYSIS
# ============================================================

elif page == "Carbon Analysis":

    st.markdown(
        '<div class="main-title">Carbon Footprint Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Translate predicted electricity consumption into an estimated carbon footprint.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        request_rate = st.slider(
            "Request rate",
            10,
            1000,
            100,
            10,
            key="carbon_request"
        )

    with c2:
        output_length = st.selectbox(
            "Output length",
            [256, 512, 1024],
            key="carbon_output"
        )

    country = st.selectbox(
        "Electricity system",
        list(carbon_intensity.keys()),
        index=list(carbon_intensity.keys()).index("India")
    )

    input_data = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    predicted_energy = float(model.predict(input_data)[0])

    intensity = carbon_intensity[country]

    emissions = predicted_energy * intensity

    st.markdown('<div class="section-title">Estimated Impact</div>',
                unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Predicted Energy",
            f"{predicted_energy:.6f} kWh"
        )

    with b:
        st.metric(
            "Grid Carbon Intensity",
            f"{intensity:.2f} gCO₂/kWh"
        )

    with c:
        st.metric(
            "Estimated Emissions",
            f"{emissions:.3f} gCO₂"
        )

    st.markdown('<div class="section-title">Calculation</div>',
                unsafe_allow_html=True)

    st.latex(
        r"\text{Carbon Emissions} = "
        r"\text{Predicted Energy} \times "
        r"\text{Grid Carbon Intensity}"
    )

    st.write(
        f"For this workload, the model predicts approximately "
        f"**{predicted_energy:.6f} kWh** of electricity consumption. "
        f"Using the selected electricity system's carbon intensity of "
        f"**{intensity:.2f} gCO₂/kWh**, the estimated carbon footprint is "
        f"**{emissions:.3f} gCO₂**."
    )

    st.caption(
        "Carbon estimates are scenario calculations based on the selected "
        "country's electricity carbon intensity."
    )


# ============================================================
# COUNTRY COMPARISON
# ============================================================

elif page == "Country Comparison":

    st.markdown(
        '<div class="main-title">Country-Level Carbon Comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Compare the estimated carbon footprint of the same AI workload across different electricity systems.</div>',
        unsafe_allow_html=True
    )

    request_rate = st.slider(
        "Request rate",
        10,
        1000,
        100,
        10,
        key="country_request"
    )

    output_length = st.selectbox(
        "Output length",
        [256, 512, 1024],
        key="country_output"
    )

    input_data = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    energy = float(model.predict(input_data)[0])

    comparison = pd.DataFrame({
        "Country": list(carbon_intensity.keys()),
        "Carbon Intensity (gCO₂/kWh)": list(carbon_intensity.values())
    })

    comparison["Estimated Emissions (gCO₂)"] = (
        energy * comparison["Carbon Intensity (gCO₂/kWh)"]
    )

    comparison = comparison.sort_values(
        "Estimated Emissions (gCO₂)"
    )

    fig = px.bar(
        comparison,
        x="Country",
        y="Estimated Emissions (gCO₂)",
        title="Estimated Carbon Footprint by Electricity System",
        text_auto=".2f"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Estimated CO₂ emissions (g)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        comparison.style.format({
            "Carbon Intensity (gCO₂/kWh)": "{:.2f}",
            "Estimated Emissions (gCO₂)": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The energy prediction is held constant across countries. The difference "
        "in estimated emissions comes from differences in electricity-grid carbon intensity."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Comparison of supervised machine-learning and deep-learning approaches.</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        model_results.style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">R² Comparison</div>',
                unsafe_allow_html=True)

    fig = px.bar(
        model_results,
        x="Model",
        y="R²",
        text_auto=".3f",
        title="Model Predictive Performance"
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="R²",
        xaxis_title="",
        yaxis_range=[0, 1],
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown('<div class="section-title">Selected Model</div>',
                unsafe_allow_html=True)

    st.write(
        """
        Gradient Boosting achieved the strongest average validation performance
        among the evaluated models. Its mean 5-fold validation R² was 0.9181,
        with an MAE of 0.002004 kWh and an RMSE of 0.002342 kWh.

        The deep-learning model was retained as an experimental comparison rather
        than the deployed predictor because its validation performance was lower
        for this dataset.
        """
    )

    st.markdown(
        '<div class="info-box">'
        '<strong>Generalisation check:</strong><br>'
        'The Gradient Boosting model achieved a pooled out-of-fold R² of '
        '<strong>0.9611</strong> under condition-grouped validation.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.markdown(
        '<div class="main-title">Model Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Understanding which workload characteristics contribute most to predicted energy consumption.</div>',
        unsafe_allow_html=True
    )

    importance = pd.DataFrame({
        "Feature": [
            "Request Rate",
            "Output Length"
        ],
        "Mean Absolute SHAP Value": [
            0.006561,
            0.003193
        ]
    })

    importance["Relative Contribution"] = (
        importance["Mean Absolute SHAP Value"] /
        importance["Mean Absolute SHAP Value"].sum()
    ) * 100

    fig = px.bar(
        importance,
        x="Feature",
        y="Mean Absolute SHAP Value",
        text_auto=".4f",
        title="Feature Influence on Energy Predictions"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Mean |SHAP value|",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        importance.style.format({
            "Mean Absolute SHAP Value": "{:.6f}",
            "Relative Contribution": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">Interpretation</div>',
                unsafe_allow_html=True)

    st.write(
        """
        Request rate has the largest average absolute SHAP value, indicating that
        it contributes more strongly to the model's predictions than output length.

        The model also captures a nonlinear relationship between request rate and
        energy consumption. In the observed workload data, energy consumption
        decreases sharply at low request rates and then approaches a plateau at
        higher request rates.
        """
    )

    st.warning(
        "Feature importance describes the model's predictive behaviour. "
        "It should not be interpreted as proof of a causal relationship."
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="main-title">Research Methodology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">End-to-end predictive modelling workflow.</div>',
        unsafe_allow_html=True
    )

    stages = [
        (
            "1. Problem Definition",
            "Predict electricity consumption of an AI inference workload."
        ),
        (
            "2. Data Understanding",
            "Analyse workload characteristics, energy measurements and experimental conditions."
        ),
        (
            "3. Data Cleaning",
            "Remove incomplete workload runs and retain completed observations."
        ),
        (
            "4. Feature Selection",
            "Use request rate and output length as the primary pre-run workload predictors."
        ),
        (
            "5. Model Development",
            "Compare Linear Regression, Ridge, Random Forest, Gradient Boosting and a neural network."
        ),
        (
            "6. Validation",
            "Use condition-grouped 5-fold GroupKFold validation to avoid train-test overlap between workload conditions."
        ),
        (
            "7. Interpretation",
            "Use SHAP-based feature importance to examine model behaviour."
        ),
        (
            "8. Carbon Extension",
            "Combine predicted energy with country-level electricity carbon intensity."
        )
    ]

    for title, description in stages:

        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:12px;">
                <strong>{title}</strong>
                <br>
                <span style="color:#6b7280;">{description}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Prediction Pipeline</div>',
                unsafe_allow_html=True)

    pipeline = pd.DataFrame({
        "Input": ["Request Rate", "Output Length"],
        "↓": ["↓", "↓"],
        "Model": ["Gradient Boosting Regressor", "Gradient Boosting Regressor"],
        "↓": ["↓", "↓"],
        "Output": ["Energy Consumption", "Energy Consumption"]
    })

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.markdown(
        '<div class="main-title">Limitations & Scope</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Important considerations when interpreting the results.</div>',
        unsafe_allow_html=True
    )

    limitations = [
        (
            "Dataset scope",
            "The dataset represents a specific AI inference workload configuration and should not automatically be generalized to every AI model or hardware environment."
        ),
        (
            "Workload characteristics",
            "The predictive model uses request rate and output length as its primary predictors. Additional hardware and system variables could improve broader modelling."
        ),
        (
            "Carbon estimation",
            "Carbon emissions are estimated by multiplying predicted energy by a country-level grid carbon intensity."
        ),
        (
            "Temporal resolution",
            "The country comparison uses annual electricity carbon-intensity values and therefore does not represent real-time hourly grid conditions."
        ),
        (
            "Causal interpretation",
            "The model identifies predictive relationships rather than proving that changing a workload variable will necessarily cause a corresponding change in energy consumption."
        ),
        (
            "Operational constraints",
            "A real-world carbon-aware deployment decision would also need to consider latency, cost, hardware availability, data privacy and workload location."
        )
    ]

    for title, description in limitations:

        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:14px;">
                <div style="font-size:18px;font-weight:600;">
                    {title}
                </div>
                <div style="color:#6b7280;margin-top:6px;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Future Improvements</div>',
                unsafe_allow_html=True)

    st.write(
        """
        Future work could incorporate GPU utilisation, memory utilisation,
        hardware configuration, execution duration, power characteristics and
        higher-resolution electricity-grid carbon intensity. These additions
        could support more comprehensive workload-level energy modelling and
        eventually enable real-time carbon-aware scheduling.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        AI Energy & Carbon Intelligence | Predictive Modelling Research Project
        <br>
        Gradient Boosting Regression | Energy Consumption Prediction | Carbon Analysis
    </div>
    """,
    unsafe_allow_html=True
)
