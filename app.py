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
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f5f7fb;
    }

    .main {
        padding-top: 1rem;
    }

    h1, h2, h3 {
        color: #172033;
    }

    p {
        color: #566176;
    }


    /* ---------- TOP NAVIGATION ---------- */

    div[data-testid="stRadio"] > div {
        background: #172033;
        padding: 8px;
        border-radius: 14px;
        gap: 5px;
    }

    div[data-testid="stRadio"] label {
        background: transparent;
        color: #dce3f0;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 600;
    }

    div[data-testid="stRadio"] label:hover {
        background: #26324a;
        color: white;
    }

    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #ffffff;
        color: #172033;
    }

    div[data-testid="stRadio"] input {
        display: none;
    }


    /* ---------- HEADER ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #172033 0%,
            #263a5b 55%,
            #315477 100%
        );
        padding: 38px 42px;
        border-radius: 22px;
        margin: 15px 0 28px 0;
        color: white;
        box-shadow: 0 10px 30px rgba(23,32,51,0.12);
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 8px;
        font-weight: 750;
    }

    .hero p {
        color: #d9e2ef;
        font-size: 16px;
        max-width: 850px;
        line-height: 1.6;
        margin-bottom: 0;
    }


    /* ---------- SECTION TITLES ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #172033;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #69758a;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ---------- KPI CARDS ---------- */

    .metric-card {
        background: white;
        padding: 22px;
        border-radius: 17px;
        border: 1px solid #e5e9f1;
        box-shadow: 0 5px 18px rgba(23,32,51,0.05);
        min-height: 125px;
    }

    .metric-label {
        color: #758096;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #172033;
        font-size: 29px;
        font-weight: 750;
        margin-top: 8px;
    }

    .metric-description {
        color: #8791a3;
        font-size: 12px;
        margin-top: 5px;
    }


    /* ---------- INFO CARDS ---------- */

    .info-card {
        background: white;
        border: 1px solid #e5e9f1;
        border-radius: 17px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(23,32,51,0.04);
    }

    .info-card h3 {
        margin-top: 0;
        color: #172033;
        font-size: 19px;
    }

    .info-card p {
        line-height: 1.65;
    }


    /* ---------- RESULT CARD ---------- */

    .prediction-card {
        background: linear-gradient(135deg, #172033, #263a5b);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(23,32,51,0.15);
    }

    .prediction-card .label {
        color: #bfcce0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-card .value {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin: 8px 0;
    }

    .prediction-card .small {
        color: #d5deeb;
        font-size: 13px;
    }


    /* ---------- BADGES ---------- */

    .badge {
        display: inline-block;
        padding: 6px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background: #eef2f7;
        color: #334155;
    }


    /* ---------- RESEARCH FINDINGS ---------- */

    .finding {
        background: white;
        border-left: 4px solid #315477;
        padding: 17px 20px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 12px;
        box-shadow: 0 3px 12px rgba(23,32,51,0.04);
    }

    .finding strong {
        color: #172033;
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #8a94a6;
        font-size: 12px;
        padding: 35px 0 15px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("ai_energy_model.pkl")


model = load_model()


# ============================================================
# CONSTANTS
# ============================================================

VALIDATION_RMSE = 0.002342
VALIDATION_MAE = 0.002004
OOF_R2 = 0.9611

carbon_intensity = {
    "Norway": 30.75,
    "France": 40.83,
    "Brazil": 106.06,
    "Canada": 185.35,
    "UK": 216.50,
    "Germany": 338.36,
    "US": 383.78,
    "Australia": 553.83,
    "China": 555.40,
    "India": 707.45
}


# ============================================================
# MODEL RESULTS
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
# AGGREGATED DATA FROM EXPERIMENT
# ============================================================

request_rates = [
    10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 200, 300, 400, 500, 600, 700,
    800, 900, 1000
]

energy_by_rate = [
    0.069743, 0.040786, 0.031892, 0.028035,
    0.026093, 0.024658, 0.024465, 0.024048,
    0.023751, 0.023547, 0.022557, 0.022232,
    0.022061, 0.021942, 0.021923, 0.021859,
    0.021858, 0.021834, 0.021784
]

energy_by_length = pd.DataFrame({
    "Output Length": [256, 512, 1024],
    "Mean Energy (kWh)": [
        0.022296,
        0.027094,
        0.031937
    ]
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_energy(request_rate, output_length):

    input_data = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    prediction = model.predict(input_data)[0]

    return max(float(prediction), 0)


def get_risk_category(prediction):

    if prediction < 0.020:
        return "Low"

    elif prediction < 0.040:
        return "Medium"

    return "High"


def prediction_interval(prediction):

    lower = max(0, prediction - VALIDATION_RMSE)
    upper = prediction + VALIDATION_RMSE

    return lower, upper


def carbon_emissions(energy, intensity):

    return energy * intensity


# ============================================================
# TOP NAVIGATION
# ============================================================

st.markdown("""
<div style="text-align:center; margin-bottom:10px;">
    <div style="
        font-size:13px;
        color:#7a8496;
        font-weight:600;
        letter-spacing:1px;
        text-transform:uppercase;">
        AI Sustainability Intelligence Platform
    </div>
</div>
""", unsafe_allow_html=True)


pages = [
    "Dashboard",
    "Project Overview",
    "Dataset",
    "Energy Prediction",
    "Reliability",
    "What-If Analysis",
    "Carbon Analysis",
    "Country Comparison",
    "Benchmarks",
    "Feature Importance",
    "Research Findings"
]


selected_page = st.radio(
    "",
    pages,
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <h1>AI Energy & Carbon Intelligence</h1>
    <p>
        A machine learning system for predicting the electricity
        consumption of AI inference workloads and evaluating their
        potential carbon impact across different electricity systems.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if selected_page == "Dashboard":

    st.markdown(
        '<div class="section-title">Research Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A high-level view of the predictive model and its key findings.'
        '</div>',
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
            <div class="metric-description">Mean 5-fold CV</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">0.00234</div>
            <div class="metric-description">kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Observations</div>
            <div class="metric-value">1,024</div>
            <div class="metric-description">Completed runs</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        fig = px.line(
            x=request_rates,
            y=energy_by_rate,
            markers=True,
            labels={
                "x": "Request Rate",
                "y": "Mean Energy (kWh)"
            },
            title="Energy Consumption vs Request Rate"
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.bar(
            model_results,
            x="Model",
            y="R²",
            title="Model Performance Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <h3>What the model does</h3>
        <p>
        The system takes two workload characteristics as inputs:
        request rate and output length. It predicts the electricity
        consumed by the AI inference workload in kilowatt-hours.
        The predicted energy value is then used as the basis for
        carbon-footprint scenario analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

elif selected_page == "Project Overview":

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understanding the research problem and prediction task.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>The Problem</h3>
        <p>
        Artificial intelligence workloads require electricity to operate.
        As AI inference becomes increasingly widespread, understanding
        and predicting its energy consumption becomes important for
        sustainable computing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>Prediction Task</h3>
            <p>
            Predict the electricity consumed by an AI inference workload
            using workload characteristics available before or during
            workload planning.
            </p>
            <p>
            <strong>Target:</strong> Energy consumption (kWh)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Predictors</h3>
            <p>
            <strong>Request Rate:</strong> incoming workload intensity.
            </p>
            <p>
            <strong>Output Length:</strong> number of tokens generated
            by the inference workload.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Research Question</h3>
        <p style="font-size:18px;">
        Can machine learning accurately predict the energy consumption
        of AI inference workloads using workload characteristics?
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

elif selected_page == "Dataset":

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Characteristics of the AI inference workload experiments.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        ("Observations", "1,024", "Completed workload runs"),
        ("Request Rates", "19", "10 to 1,000 requests"),
        ("Output Lengths", "3", "256, 512, 1,024 tokens"),
        ("Target", "Energy", "Measured in kWh")
    ]

    for col, stat in zip([c1, c2, c3, c4], stats):

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{stat[0]}</div>
                <div class="metric-value">{stat[1]}</div>
                <div class="metric-description">{stat[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        fig = px.line(
            x=request_rates,
            y=energy_by_rate,
            markers=True,
            labels={
                "x": "Request Rate",
                "y": "Mean Energy (kWh)"
            },
            title="Energy Consumption Across Request Rates"
        )

        fig.update_layout(
            template="plotly_white",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.bar(
            energy_by_length,
            x="Output Length",
            y="Mean Energy (kWh)",
            title="Energy Consumption by Output Length"
        )

        fig.update_layout(
            template="plotly_white",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <h3>Observed Energy Distribution</h3>
        <p>
        The completed runs have a mean energy consumption of approximately
        0.0272 kWh. Energy values range from approximately 0.0143 kWh to
        0.0774 kWh.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ENERGY PREDICTION
# ============================================================

elif selected_page == "Energy Prediction":

    st.markdown(
        '<div class="section-title">Energy Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter workload characteristics to estimate electricity consumption.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1.4])

    with left:

        request_rate = st.slider(
            "Request Rate",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

        output_length = st.selectbox(
            "Output Length",
            [256, 512, 1024],
            index=1
        )

        predict_button = st.button(
            "Predict Energy Consumption",
            use_container_width=True
        )

    with right:

        prediction = predict_energy(
            request_rate,
            output_length
        )

        risk = get_risk_category(prediction)

        st.markdown(f"""
        <div class="prediction-card">
            <div class="label">Predicted Energy Consumption</div>
            <div class="value">{prediction:.5f} kWh</div>
            <div class="small">
                Risk category: {risk}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Estimated MAE",
                f"{VALIDATION_MAE:.5f} kWh"
            )

        with c2:
            st.metric(
                "Validation RMSE",
                f"{VALIDATION_RMSE:.5f} kWh"
            )

    st.markdown("""
    <div class="info-card">
        <h3>Interpretation</h3>
        <p>
        The prediction represents the estimated electricity consumption
        associated with the selected AI workload configuration. The model
        was trained using request rate and output length as predictors.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RELIABILITY
# ============================================================

elif selected_page == "Reliability":

    st.markdown(
        '<div class="section-title">Prediction Reliability</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'An error-based view of the model prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    request_rate = st.slider(
        "Request Rate",
        10,
        1000,
        100,
        10,
        key="reliability_rate"
    )

    output_length = st.selectbox(
        "Output Length",
        [256, 512, 1024],
        key="reliability_length"
    )

    prediction = predict_energy(
        request_rate,
        output_length
    )

    lower, upper = prediction_interval(prediction)

    risk = get_risk_category(prediction)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted Energy",
            f"{prediction:.5f} kWh"
        )

    with c2:
        st.metric(
            "Approx. Lower Bound",
            f"{lower:.5f} kWh"
        )

    with c3:
        st.metric(
            "Approx. Upper Bound",
            f"{upper:.5f} kWh"
        )

    st.markdown(f"""
    <div class="info-card">
        <h3>Energy Risk Category: {risk}</h3>
        <p>
        This category is an analytical indicator based on the predicted
        energy value. It is not a probability of model correctness.
        </p>
        <p>
        The approximate error range is based on the model's validation
        RMSE of {VALIDATION_RMSE:.5f} kWh.
        </p>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=["Lower estimate", "Prediction", "Upper estimate"],
            y=[lower, prediction, upper],
            mode="lines+markers",
            line=dict(width=4)
        )
    )

    fig.update_layout(
        title="Prediction Error Range",
        template="plotly_white",
        yaxis_title="Energy (kWh)",
        height=380
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

elif selected_page == "What-If Analysis":

    st.markdown(
        '<div class="section-title">What-If Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore how different workload configurations affect predicted energy.'
        '</div>',
        unsafe_allow_html=True
    )

    output_length = st.selectbox(
        "Choose Output Length",
        [256, 512, 1024],
        index=1
    )

    scenario_rates = np.array(
        [10, 20, 50, 100, 200, 400, 600, 800, 1000]
    )

    scenario_predictions = [
        predict_energy(rate, output_length)
        for rate in scenario_rates
    ]

    scenario_df = pd.DataFrame({
        "Request Rate": scenario_rates,
        "Predicted Energy (kWh)": scenario_predictions
    })

    fig = px.line(
        scenario_df,
        x="Request Rate",
        y="Predicted Energy (kWh)",
        markers=True,
        title=f"Predicted Energy Across Request Rates — {output_length} Tokens"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        scenario_df.style.format({
            "Predicted Energy (kWh)": "{:.5f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CARBON ANALYSIS
# ============================================================

elif selected_page == "Carbon Analysis":

    st.markdown(
        '<div class="section-title">Carbon Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Convert predicted electricity consumption into estimated CO₂ emissions.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        request_rate = st.slider(
            "Request Rate",
            10,
            1000,
            100,
            10,
            key="carbon_rate"
        )

        output_length = st.selectbox(
            "Output Length",
            [256, 512, 1024],
            key="carbon_length"
        )

        country = st.selectbox(
            "Electricity System",
            list(carbon_intensity.keys())
        )

    prediction = predict_energy(
        request_rate,
        output_length
    )

    intensity = carbon_intensity[country]

    emissions = carbon_emissions(
        prediction,
        intensity
    )

    with right:

        st.markdown(f"""
        <div class="prediction-card">
            <div class="label">Estimated Carbon Footprint</div>
            <div class="value">{emissions:.3f} gCO₂</div>
            <div class="small">
                Based on predicted energy of {prediction:.5f} kWh
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Calculation</h3>
        <p>
        Estimated emissions are calculated by multiplying predicted
        electricity consumption by the selected electricity system's
        carbon intensity.
        </p>
        <p>
        <strong>Energy × Carbon Intensity = Estimated CO₂</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# COUNTRY COMPARISON
# ============================================================

elif selected_page == "Country Comparison":

    st.markdown(
        '<div class="section-title">Country Comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Compare the estimated carbon footprint of the same AI workload '
        'under different electricity systems.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        request_rate = st.slider(
            "Request Rate",
            10,
            1000,
            100,
            10,
            key="country_rate"
        )

    with col2:

        output_length = st.selectbox(
            "Output Length",
            [256, 512, 1024],
            key="country_length"
        )

    prediction = predict_energy(
        request_rate,
        output_length
    )

    comparison = pd.DataFrame({
        "Country": list(carbon_intensity.keys()),
        "Carbon Intensity": list(carbon_intensity.values())
    })

    comparison["Estimated CO₂ (g)"] = (
        prediction * comparison["Carbon Intensity"]
    )

    fig = px.bar(
        comparison.sort_values("Estimated CO₂ (g)"),
        x="Estimated CO₂ (g)",
        y="Country",
        orientation="h",
        title=f"Estimated Carbon Footprint — {output_length} Tokens"
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Estimated CO₂ (g)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comparison.style.format({
            "Carbon Intensity": "{:.2f}",
            "Estimated CO₂ (g)": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BENCHMARKS
# ============================================================

elif selected_page == "Benchmarks":

    st.markdown(
        '<div class="section-title">Model Benchmarks & Validation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Comparison of regression and deep learning models.'
        '</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        model_results.style.format({
            "MAE": "{:.5f}",
            "RMSE": "{:.5f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    left, right = st.columns(2)

    with left:

        fig = px.bar(
            model_results,
            x="Model",
            y="R²",
            title="R² Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            height=420,
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.bar(
            model_results,
            x="Model",
            y="RMSE",
            title="RMSE Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <h3>Selected Model: Gradient Boosting Regressor</h3>
        <p>
        Gradient Boosting achieved the strongest overall validation
        performance among the tested models, with a mean cross-validation
        R² of 0.9181 and RMSE of 0.002342 kWh.
        </p>
        <p>
        The pooled out-of-fold R² was approximately 0.9611.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif selected_page == "Feature Importance":

    st.markdown(
        '<div class="section-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'SHAP-based interpretation of the selected Gradient Boosting model.'
        '</div>',
        unsafe_allow_html=True
    )

    shap_values = pd.DataFrame({
        "Feature": [
            "Request Rate",
            "Output Length"
        ],
        "Mean |SHAP|": [
            0.006561,
            0.003193
        ]
    })

    fig = px.bar(
        shap_values.sort_values("Mean |SHAP|"),
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Mean Absolute SHAP Importance"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    ratio = 0.006561 / 0.003193

    st.markdown(f"""
    <div class="info-card">
        <h3>Interpretation</h3>
        <p>
        Request rate is the most influential predictor in the final model,
        with a mean absolute SHAP value approximately {ratio:.1f} times
        that of output length.
        </p>
        <p>
        This indicates that workload intensity plays a stronger role in
        the model's energy predictions within this experimental dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# RESEARCH FINDINGS
# ============================================================

elif selected_page == "Research Findings":

    st.markdown(
        '<div class="section-title">Research Findings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Key empirical findings from the modelling experiment.'
        '</div>',
        unsafe_allow_html=True
    )

    findings = [

        (
            "Gradient Boosting performed best.",
            "The Gradient Boosting model achieved a mean 5-fold "
            "cross-validation R² of 0.9181 and RMSE of 0.002342 kWh."
        ),

        (
            "Request rate was the strongest predictor.",
            "SHAP analysis showed request rate had substantially greater "
            "average influence than output length."
        ),

        (
            "Output length affects energy consumption.",
            "Average energy increased from approximately 0.0223 kWh "
            "for 256 tokens to 0.0319 kWh for 1,024 tokens."
        ),

        (
            "Energy does not increase proportionally with output length.",
            "The relationship between output length and energy consumption "
            "is not simply linear across the tested workloads."
        ),

        (
            "Carbon intensity changes the environmental footprint.",
            "The same predicted workload can have substantially different "
            "estimated emissions depending on the electricity system "
            "supplying the computation."
        ),

        (
            "The model is workload-specific.",
            "The findings describe the experimental AI inference workload "
            "represented in the dataset and should not automatically be "
            "generalised to every AI model or data centre."
        )
    ]

    for title, description in findings:

        st.markdown(f"""
        <div class="finding">
            <strong>{title}</strong>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    AI Energy & Carbon Intelligence · Machine Learning Research Project
</div>
""", unsafe_allow_html=True)
