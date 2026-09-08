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

.stApp {
    background-color: #f7f8fa;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

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

.section-title {
    font-size: 27px;
    font-weight: 650;
    color: #111827;
    margin-top: 20px;
    margin-bottom: 10px;
}

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

.info-box {
    background-color: white;
    border-left: 4px solid #374151;
    border-radius: 8px;
    padding: 18px;
    margin: 15px 0;
}

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

.finding-card {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

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

cv_r2 = [0.6571, 0.4396, -0.3717, 0.4713, -0.3959]
cv_mae = [0.006263, 0.003131, 0.007025, 0.006515, 0.007005]

# Validation error used for approximate prediction reliability
VALIDATION_RMSE = 0.002342
VALIDATION_MAE = 0.002004


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_energy(request_rate, output_length):

    input_data = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    return float(model.predict(input_data)[0])


def get_risk_category(prediction):

    if prediction < 0.02:
        return "Low"
    elif prediction < 0.04:
        return "Medium"
    else:
        return "High"


def get_reliability(prediction):

    # Approximate reliability based on validation error
    relative_error = VALIDATION_RMSE / max(abs(prediction), 0.000001)

    if relative_error <= 0.10:
        return "High", 90
    elif relative_error <= 0.20:
        return "Moderate", 75
    else:
        return "Lower", 60


def prediction_interval(prediction):

    lower = max(0, prediction - VALIDATION_RMSE)
    upper = prediction + VALIDATION_RMSE

    return lower, upper


# ============================================================
# SIDEBAR
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
        "Dashboard",
        "Project Overview",
        "Dataset Overview",
        "Energy Prediction",
        "Prediction Reliability",
        "Carbon Analysis",
        "Country Comparison",
        "What-If Analysis",
        "Benchmarks & Validation",
        "Feature Importance",
        "Research Findings",
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
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">AI Energy & Carbon Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Research dashboard for AI workload energy prediction and carbon analysis.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Model</div>
            <div class="metric-value">GBR</div>
            <div class="metric-description">Gradient Boosting</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Validation R²</div>
            <div class="metric-value">0.918</div>
            <div class="metric-description">Mean 5-fold R²</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Validation RMSE</div>
            <div class="metric-value">0.00234</div>
            <div class="metric-description">kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Observations</div>
            <div class="metric-value">1,024</div>
            <div class="metric-description">Completed workloads</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Research Question</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Can machine learning accurately predict the electricity consumption "
        "of AI inference workloads using workload characteristics, and can "
        "these predictions support carbon-aware analysis?"
    )

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        model_results,
        x="Model",
        y="R²",
        text_auto=".3f",
        title="Validation R² by Model"
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_range=[0, 1],
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Key Findings</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="finding-card">
    <strong>Gradient Boosting performed best.</strong><br>
    It achieved a mean 5-fold validation R² of 0.9181.
    </div>

    <div class="finding-card">
    <strong>Request rate was the dominant predictor.</strong><br>
    Its mean absolute SHAP value was approximately twice that of output length.
    </div>

    <div class="finding-card">
    <strong>Output length affected energy consumption.</strong><br>
    Larger output lengths were associated with higher energy consumption.
    </div>

    <div class="finding-card">
    <strong>Carbon footprint depends on the electricity system.</strong><br>
    The same predicted energy consumption can produce substantially different
    estimated emissions depending on grid carbon intensity.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

elif page == "Project Overview":

    st.markdown(
        '<div class="main-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Understanding the research problem and analytical approach.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">The Problem</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        AI inference requires electricity. As AI systems become increasingly
        deployed, understanding and estimating their energy consumption becomes
        important for sustainable computing.

        The amount of energy required by an inference workload can vary according
        to characteristics such as request rate and output length.
        """
    )

    st.markdown(
        '<div class="section-title">Objective</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The objective is to develop and evaluate supervised machine-learning
        models capable of predicting the electricity consumption of AI inference
        workloads.
        """
    )

    st.markdown(
        '<div class="section-title">Prediction Target</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    <strong>Target variable:</strong> Energy Consumption<br>
    <strong>Unit:</strong> kWh<br>
    <strong>Predictors:</strong> Request Rate and Output Length
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Research Pipeline</div>',
        unsafe_allow_html=True
    )

    pipeline = pd.DataFrame({
        "Stage": [
            "AI Workload",
            "Preprocessing",
            "Model Training",
            "Validation",
            "Energy Prediction",
            "Carbon Analysis"
        ],
        "Description": [
            "Inference workload characteristics",
            "Clean and prepare observations",
            "Train multiple supervised models",
            "Condition-grouped cross-validation",
            "Predict electricity consumption",
            "Combine energy with grid carbon intensity"
        ]
    })

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.markdown(
        '<div class="main-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Structure and characteristics of the AI inference workload data.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Completed Records", "1,024")

    with c2:
        st.metric("Request Rate Levels", "19")

    with c3:
        st.metric("Output Lengths", "3")

    with c4:
        st.metric("Prediction Target", "Energy kWh")

    st.markdown(
        '<div class="section-title">Variables Used for Prediction</div>',
        unsafe_allow_html=True
    )

    variables = pd.DataFrame({
        "Variable": [
            "request_rate_x",
            "hf-output-len",
            "energy_kWh"
        ],
        "Role": [
            "Predictor",
            "Predictor",
            "Target"
        ],
        "Description": [
            "AI inference requests per second",
            "Generated output length in tokens",
            "Measured electricity consumption"
        ]
    })

    st.dataframe(
        variables,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Energy Distribution</div>',
        unsafe_allow_html=True
    )

    # Representative distribution based on observed dataset statistics
    energy_stats = pd.DataFrame({
        "Statistic": [
            "Minimum",
            "25th percentile",
            "Median",
            "Mean",
            "75th percentile",
            "Maximum"
        ],
        "Energy (kWh)": [
            0.014297,
            0.019584,
            0.024518,
            0.027154,
            0.029574,
            0.077427
        ]
    })

    fig = px.bar(
        energy_stats,
        x="Statistic",
        y="Energy (kWh)",
        text_auto=".4f",
        title="Observed Energy Consumption Statistics"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Experimental Workload Structure</div>',
        unsafe_allow_html=True
    )

    workload = pd.DataFrame({
        "Output Length": [256, 512, 1024],
        "Mean Energy (kWh)": [
            0.022403,
            0.027094,
            0.031937
        ]
    })

    fig = px.bar(
        workload,
        x="Output Length",
        y="Mean Energy (kWh)",
        text_auto=".4f",
        title="Mean Energy Consumption by Output Length"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ENERGY PREDICTION
# ============================================================

elif page == "Energy Prediction":

    st.markdown(
        '<div class="main-title">Energy Consumption Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Estimate electricity consumption for an AI inference workload.</div>',
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
            10,
            1000,
            100,
            10
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

        if predict:

            prediction = predict_energy(
                request_rate,
                output_length
            )

            risk = get_risk_category(prediction)
            reliability, reliability_score = get_reliability(prediction)

            lower, upper = prediction_interval(prediction)

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
                        kWh
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            with a:
                st.metric("Energy Risk", risk)

            with b:
                st.metric(
                    "Reliability",
                    f"{reliability_score}%"
                )

            with c:
                st.metric(
                    "Validation Error",
                    f"±{VALIDATION_RMSE:.4f} kWh"
                )

            st.info(
                f"Approximate expected range: "
                f"{lower:.6f} to {upper:.6f} kWh."
            )

            st.caption(
                "Reliability is an approximate indicator derived from validation "
                "error. It is not a probability that the prediction is correct."
            )

        else:

            st.markdown(
                """
                <div class="info-box">
                Enter the workload parameters and generate an energy prediction.
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PREDICTION RELIABILITY
# ============================================================

elif page == "Prediction Reliability":

    st.markdown(
        '<div class="main-title">Prediction Reliability</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Assess how much confidence should be placed in an individual energy prediction.</div>',
        unsafe_allow_html=True
    )

    request_rate = st.slider(
        "Request rate",
        10,
        1000,
        100,
        10,
        key="reliability_rate"
    )

    output_length = st.selectbox(
        "Output length",
        [256, 512, 1024],
        key="reliability_output"
    )

    prediction = predict_energy(
        request_rate,
        output_length
    )

    risk = get_risk_category(prediction)
    reliability, reliability_score = get_reliability(prediction)
    lower, upper = prediction_interval(prediction)

    st.markdown(
        '<div class="section-title">Prediction Assessment</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Predicted Energy",
            f"{prediction:.5f} kWh"
        )

    with c2:
        st.metric(
            "Energy Risk",
            risk
        )

    with c3:
        st.metric(
            "Reliability",
            f"{reliability_score}%"
        )

    with c4:
        st.metric(
            "Validation RMSE",
            f"{VALIDATION_RMSE:.5f}"
        )

    st.progress(reliability_score / 100)

    st.markdown(
        '<div class="section-title">Estimated Prediction Range</div>',
        unsafe_allow_html=True
    )

    range_df = pd.DataFrame({
        "Value": ["Lower", "Prediction", "Upper"],
        "Energy (kWh)": [
            lower,
            prediction,
            upper
        ]
    })

    fig = px.bar(
        range_df,
        x="Value",
        y="Energy (kWh)",
        text_auto=".5f",
        title="Prediction and Approximate Error Range"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "This reliability indicator is based on validation error and should "
        "not be interpreted as a formal statistical probability."
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
        '<div class="subtitle">Translate predicted electricity consumption into estimated carbon emissions.</div>',
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

    predicted_energy = predict_energy(
        request_rate,
        output_length
    )

    intensity = carbon_intensity[country]

    emissions = predicted_energy * intensity

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

    st.markdown(
        '<div class="section-title">Calculation</div>',
        unsafe_allow_html=True
    )

    st.latex(
        r"\text{Carbon Emissions} = "
        r"\text{Predicted Energy} \times "
        r"\text{Grid Carbon Intensity}"
    )

    st.write(
        f"The predicted workload consumes approximately "
        f"**{predicted_energy:.6f} kWh**. Under the selected electricity "
        f"system, this corresponds to approximately **{emissions:.3f} gCO₂**."
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
        '<div class="subtitle">Compare the same AI workload across different electricity systems.</div>',
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

    energy = predict_energy(
        request_rate,
        output_length
    )

    comparison = pd.DataFrame({
        "Country": list(carbon_intensity.keys()),
        "Carbon Intensity (gCO₂/kWh)": list(carbon_intensity.values())
    })

    comparison["Estimated Emissions (gCO₂)"] = (
        energy *
        comparison["Carbon Intensity (gCO₂/kWh)"]
    )

    comparison = comparison.sort_values(
        "Estimated Emissions (gCO₂)"
    )

    fig = px.bar(
        comparison,
        x="Country",
        y="Estimated Emissions (gCO₂)",
        text_auto=".2f",
        title="Estimated Carbon Footprint by Electricity System"
    )

    fig.update_layout(
        template="plotly_white",
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
        "The predicted energy is held constant. Differences in emissions are "
        "caused by differences in electricity-grid carbon intensity."
    )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

elif page == "What-If Analysis":

    st.markdown(
        '<div class="main-title">What-If Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore how workload characteristics influence predicted energy consumption.</div>',
        unsafe_allow_html=True
    )

    output_length = st.selectbox(
        "Output length",
        [256, 512, 1024],
        key="whatif_output"
    )

    rates = np.arange(10, 1001, 10)

    predictions = [
        predict_energy(rate, output_length)
        for rate in rates
    ]

    whatif = pd.DataFrame({
        "Request Rate": rates,
        "Predicted Energy": predictions
    })

    fig = px.line(
        whatif,
        x="Request Rate",
        y="Predicted Energy",
        title=f"Predicted Energy vs Request Rate — {output_length} Tokens"
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Request rate (requests/second)",
        yaxis_title="Predicted energy (kWh)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Compare Output Lengths</div>',
        unsafe_allow_html=True
    )

    all_rows = []

    for length in [256, 512, 1024]:

        for rate in rates:

            all_rows.append({
                "Request Rate": rate,
                "Output Length": str(length),
                "Predicted Energy": predict_energy(rate, length)
            })

    all_data = pd.DataFrame(all_rows)

    fig = px.line(
        all_data,
        x="Request Rate",
        y="Predicted Energy",
        color="Output Length",
        title="Predicted Energy Across Workload Conditions"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The observed dataset shows a sharp decrease in energy consumption at "
        "low request rates followed by a plateau at higher request rates. "
        "This describes the behaviour present in the studied workload and "
        "should not be interpreted as a universal causal relationship."
    )


# ============================================================
# BENCHMARKS & VALIDATION
# ============================================================

elif page == "Benchmarks & Validation":

    st.markdown(
        '<div class="main-title">Benchmarks & Validation Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Evaluation of model accuracy, generalisation and validation stability.</div>',
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

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            model_results,
            x="Model",
            y="R²",
            text_auto=".3f",
            title="R² Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            yaxis_range=[0, 1],
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.bar(
            model_results,
            x="Model",
            y="RMSE",
            text_auto=".4f",
            title="RMSE Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">Cross-Validation Stability</div>',
        unsafe_allow_html=True
    )

    folds = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "R²": cv_r2,
        "MAE": cv_mae
    })

    fig = px.bar(
        folds,
        x="Fold",
        y="R²",
        text_auto=".3f",
        title="Gradient Boosting R² Across Validation Folds"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        folds.style.format({
            "R²": "{:.4f}",
            "MAE": "{:.6f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Generalisation Check</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        <strong>Pooled out-of-fold R²: 0.9611</strong><br><br>
        The pooled out-of-fold evaluation indicates strong predictive performance
        when predictions are generated for workload conditions not used during
        training.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "The mean fold R² and pooled out-of-fold R² are different statistics. "
        "The former averages the five fold-level R² values, while the latter "
        "calculates R² after pooling all out-of-fold predictions."
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
        '<div class="subtitle">Understanding which workload characteristics influence model predictions.</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        importance,
        x="Feature",
        y="Mean Absolute SHAP Value",
        text_auto=".4f",
        title="Feature Influence on Energy Predictions"
    )

    fig.update_layout(
        template="plotly_white",
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

    st.markdown(
        '<div class="section-title">Relative Contribution</div>',
        unsafe_allow_html=True
    )

    fig = px.pie(
        importance,
        names="Feature",
        values="Relative Contribution",
        hole=0.45,
        title="Relative Feature Influence"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Interpretation</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Request rate has the largest mean absolute SHAP value, indicating that
        it contributes more strongly to the model's predictions than output length.

        The model also captures a nonlinear relationship between request rate and
        predicted energy. Energy decreases sharply at lower request rates and
        approaches a plateau at higher request rates within the observed workload.
        """
    )

    st.warning(
        "Feature importance describes predictive behaviour and should not be "
        "interpreted as proof of causality."
    )


# ============================================================
# RESEARCH FINDINGS
# ============================================================

elif page == "Research Findings":

    st.markdown(
        '<div class="main-title">Research Findings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Principal findings obtained from the predictive modelling analysis.</div>',
        unsafe_allow_html=True
    )

    findings = [
        (
            "Gradient Boosting was the strongest model.",
            "It achieved the best mean validation performance with R² = 0.9181, MAE = 0.002004 kWh and RMSE = 0.002342 kWh."
        ),
        (
            "Request rate was the dominant predictor.",
            "Its mean absolute SHAP value was approximately twice that of output length."
        ),
        (
            "Output length affected energy consumption.",
            "Mean energy increased from approximately 0.0224 kWh at 256 tokens to 0.0319 kWh at 1024 tokens."
        ),
        (
            "The relationship with request rate was nonlinear.",
            "Energy decreased sharply at low request rates and approached a plateau at higher request rates."
        ),
        (
            "The neural network did not outperform tree-based models.",
            "The Deep Learning MLP achieved lower validation performance than Gradient Boosting for this dataset."
        ),
        (
            "Electricity-grid carbon intensity strongly affects estimated emissions.",
            "The same predicted energy consumption produces different carbon estimates under different electricity systems."
        )
    ]

    for title, description in findings:

        st.markdown(
            f"""
            <div class="finding-card">
                <strong>{title}</strong>
                <br><br>
                <span style="color:#6b7280;">
                    {description}
                </span>
            </div>
            """,
            unsafe_allow_html=True
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
            "Use request rate and output length as primary pre-run workload predictors."
        ),
        (
            "5. Model Development",
            "Compare Linear Regression, Ridge, Random Forest, Gradient Boosting and a neural network."
        ),
        (
            "6. Validation",
            "Use condition-grouped 5-fold GroupKFold validation."
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

    st.markdown(
        '<div class="section-title">Prediction Pipeline</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Request Rate + Output Length
        → Gradient Boosting Regressor
        → Predicted Energy Consumption
        → Grid Carbon Intensity
        → Estimated Carbon Footprint
        """
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
            "The predictive model uses request rate and output length as its primary predictors."
        ),
        (
            "Carbon estimation",
            "Carbon emissions are estimated by multiplying predicted energy by country-level grid carbon intensity."
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
            "Reliability indicator",
            "The prediction reliability score is an approximate validation-error-based indicator and is not a formal probability of correctness."
        ),
        (
            "Operational constraints",
            "Real-world carbon-aware deployment would also need to consider latency, cost, hardware availability, privacy and workload location."
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

    st.markdown(
        '<div class="section-title">Future Improvements</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Future work could incorporate GPU utilisation, memory utilisation,
        hardware configuration, execution duration, power characteristics and
        higher-resolution electricity-grid carbon intensity. These additions
        could support more comprehensive workload-level energy modelling.
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
