import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Carbon Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f8fa;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

h2 {
    margin-top: 1.5rem;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e6e8eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.metric-label {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
}

.info-box {
    background: white;
    border: 1px solid #e6e8eb;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.small-text {
    color: #6b7280;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ai_energy_dataset.csv")

    # Only completed runs
    if "run_completed" in df.columns:
        df = df[df["run_completed"] == True].copy()

    df = df.dropna(
        subset=[
            "request_rate_x",
            "hf-output-len",
            "energy_kWh"
        ]
    )

    return df


df = load_data()


# ============================================================
# PREPARE MODEL DATA
# ============================================================

FEATURES = [
    "request_rate_x",
    "hf-output-len"
]

TARGET = "energy_kWh"

X = df[FEATURES].copy()
y = df[TARGET].copy()

groups = (
    X["request_rate_x"].astype(str)
    + "_"
    + X["hf-output-len"].astype(str)
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

@st.cache_resource
def train_final_model(X, y):

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_final_model(X, y)


# ============================================================
# CROSS VALIDATION
# ============================================================

@st.cache_data
def run_validation(X, y, groups):

    gkf = GroupKFold(n_splits=5)

    oof_predictions = np.zeros(len(y))
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups),
        start=1
    ):

        fold_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        fold_model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions = fold_model.predict(
            X.iloc[test_idx]
        )

        oof_predictions[test_idx] = predictions

        mae = mean_absolute_error(
            y.iloc[test_idx],
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y.iloc[test_idx],
                predictions
            )
        )

        r2 = r2_score(
            y.iloc[test_idx],
            predictions
        )

        fold_results.append({
            "Fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

    fold_df = pd.DataFrame(fold_results)

    overall_mae = mean_absolute_error(
        y,
        oof_predictions
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y,
            oof_predictions
        )
    )

    overall_r2 = r2_score(
        y,
        oof_predictions
    )

    return (
        fold_df,
        oof_predictions,
        overall_mae,
        overall_rmse,
        overall_r2
    )


(
    fold_df,
    oof_predictions,
    oof_mae,
    oof_rmse,
    oof_r2
) = run_validation(X, y, groups)


# ============================================================
# VALIDATION DATA
# ============================================================

validation_df = df.copy()

validation_df["Actual"] = y.values
validation_df["Predicted"] = oof_predictions

validation_df["Error"] = (
    validation_df["Predicted"]
    - validation_df["Actual"]
)

validation_df["Absolute_Error"] = (
    validation_df["Error"].abs()
)

validation_df["Percentage_Error"] = (
    validation_df["Absolute_Error"]
    / validation_df["Actual"]
    * 100
)


# ============================================================
# MODEL BENCHMARK DATA
# ============================================================

benchmark_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest",
        "Gradient Boosting",
        "Neural Network"
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
# CARBON DATA
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
# SIDEBAR
# ============================================================

st.sidebar.title("AI Carbon Intelligence")

st.sidebar.caption(
    "Predictive analytics for AI workload energy consumption"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Project Overview",
        "Dataset Overview",
        "Energy Prediction",
        "Prediction Reliability",
        "Benchmarks & Validation",
        "Feature Importance",
        "Energy Analysis",
        "Carbon Analysis",
        "Country Comparison",
        "What-If Analysis",
        "Research Findings",
        "Methodology & Limitations"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Model: Gradient Boosting Regressor"
)

st.sidebar.caption(
    "Target: Energy Consumption (kWh)"
)


# ============================================================
# HELPER
# ============================================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("AI Carbon Intelligence")

    st.write(
        "An interactive predictive analytics platform for estimating "
        "AI inference energy consumption and exploring its carbon implications."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Valid Workload Runs",
            f"{len(df):,}"
        )

    with c2:
        metric_card(
            "Mean Energy",
            f"{df[TARGET].mean():.4f} kWh"
        )

    with c3:
        metric_card(
            "Validation R²",
            f"{oof_r2:.3f}"
        )

    with c4:
        metric_card(
            "Validation RMSE",
            f"{oof_rmse:.4f} kWh"
        )

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            validation_df,
            x="Actual",
            y="Predicted",
            title="Actual vs Predicted Energy",
            labels={
                "Actual": "Actual Energy (kWh)",
                "Predicted": "Predicted Energy (kWh)"
            }
        )

        min_val = min(
            validation_df["Actual"].min(),
            validation_df["Predicted"].min()
        )

        max_val = max(
            validation_df["Actual"].max(),
            validation_df["Predicted"].max()
        )

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            validation_df,
            x="Error",
            nbins=35,
            title="Prediction Error Distribution",
            labels={
                "Error": "Prediction Error (kWh)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Key Findings")

    st.info(
        "Gradient Boosting achieved the strongest overall validation "
        "performance among the evaluated models. Request rate was the "
        "most influential predictor, while output length also had a "
        "substantial effect on energy consumption."
    )


# ============================================================
# PROJECT OVERVIEW
# ============================================================

elif page == "Project Overview":

    st.title("Project Overview")

    st.subheader("Research Problem")

    st.write(
        "Artificial intelligence workloads consume electricity during "
        "training and inference. The amount of energy required depends "
        "on characteristics of the workload and execution conditions."
    )

    st.subheader("Research Objective")

    st.write(
        "The objective is to develop a supervised machine learning model "
        "that predicts the electricity consumption of an AI inference workload."
    )

    st.subheader("Research Question")

    st.write(
        "Can machine learning accurately predict the energy consumption "
        "of AI inference workloads using workload characteristics, and can "
        "these predictions support carbon-aware decision making?"
    )

    st.subheader("Prediction Pipeline")

    st.markdown("""
    **AI Workload**

    ↓

    **Workload Characteristics**

    Request Rate + Output Length

    ↓

    **Machine Learning Model**

    Gradient Boosting Regressor

    ↓

    **Predicted Energy Consumption**

    Energy in kWh

    ↓

    **Carbon Estimation**

    Energy × Grid Carbon Intensity

    ↓

    **Scenario Analysis**

    Compare countries and workload configurations
    """)


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.title("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Observations",
            f"{len(df):,}"
        )

    with c2:
        metric_card(
            "Input Features",
            "2"
        )

    with c3:
        metric_card(
            "Request Rate Levels",
            f"{df['request_rate_x'].nunique()}"
        )

    with c4:
        metric_card(
            "Output Lengths",
            f"{df['hf-output-len'].nunique()}"
        )

    st.subheader("Variables Used")

    variable_table = pd.DataFrame({
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
            "AI request rate",
            "Requested output length",
            "Energy consumed by workload"
        ]
    })

    st.dataframe(
        variable_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        df[
            [
                "request_rate_x",
                "hf-output-len",
                "energy_kWh"
            ]
        ].describe(),
        use_container_width=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# ENERGY PREDICTION
# ============================================================

elif page == "Energy Prediction":

    st.title("Energy Prediction")

    st.write(
        "Enter the characteristics of an AI inference workload."
    )

    col1, col2 = st.columns(2)

    with col1:

        request_rate = st.slider(
            "Request Rate",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

    with col2:

        output_length = st.selectbox(
            "Output Length",
            sorted(
                df["hf-output-len"].unique()
            )
        )

    input_df = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    prediction = model.predict(
        input_df
    )[0]

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Predicted Energy",
            f"{prediction:.4f} kWh"
        )

    with c2:
        metric_card(
            "Energy (Wh)",
            f"{prediction * 1000:.2f} Wh"
        )

    with c3:

        if prediction < 0.02:
            risk = "Low"
        elif prediction < 0.04:
            risk = "Medium"
        else:
            risk = "High"

        metric_card(
            "Energy Risk",
            risk
        )

    st.subheader("Prediction Interpretation")

    st.write(
        f"For a workload with a request rate of {request_rate} "
        f"requests and an output length of {output_length} tokens, "
        f"the model estimates approximately {prediction:.4f} kWh "
        f"of electricity consumption."
    )


# ============================================================
# PREDICTION RELIABILITY
# ============================================================

elif page == "Prediction Reliability":

    st.title("Prediction Reliability")

    st.write(
        "Because this is a regression problem, the model does not "
        "produce a classification probability. Instead, reliability "
        "is estimated using errors observed during cross-validation."
    )

    request_rate = st.slider(
        "Request Rate",
        10,
        1000,
        100,
        10
    )

    output_length = st.selectbox(
        "Output Length",
        sorted(
            df["hf-output-len"].unique()
        )
    )

    input_df = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    prediction = model.predict(
        input_df
    )[0]

    # Error statistics
    median_error = validation_df[
        "Absolute_Error"
    ].median()

    upper_error = validation_df[
        "Absolute_Error"
    ].quantile(0.90)

    lower_bound = max(
        0,
        prediction - upper_error
    )

    upper_bound = (
        prediction + upper_error
    )

    # Reliability based on relative position of error
    relative_error = (
        median_error / max(prediction, 1e-9)
    )

    reliability = max(
        0,
        min(
            100,
            100 * (1 - relative_error)
        )
    )

    if prediction < 0.02:
        risk = "Low"
    elif prediction < 0.04:
        risk = "Medium"
    else:
        risk = "High"

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Predicted Energy",
            f"{prediction:.4f} kWh"
        )

    with c2:
        metric_card(
            "Energy Risk",
            risk
        )

    with c3:
        metric_card(
            "Reliability Indicator",
            f"{reliability:.1f}%"
        )

    st.subheader("Expected Prediction Range")

    st.write(
        f"Based on the model's observed validation error distribution, "
        f"the prediction is approximately within "
        f"{lower_bound:.4f}–{upper_bound:.4f} kWh for a broad error range."
    )

    st.warning(
        "This reliability indicator is not a statistical probability "
        "that the prediction is correct. It is an error-based reliability "
        "indicator derived from validation performance."
    )

    st.subheader("Reliability by Energy-Risk Category")

    reliability_df = validation_df.copy()

    reliability_df["Risk Category"] = pd.cut(
        reliability_df["Actual"],
        bins=[
            -np.inf,
            0.02,
            0.04,
            np.inf
        ],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    risk_summary = (
        reliability_df
        .groupby(
            "Risk Category",
            observed=False
        )
        .agg(
            MAE=("Absolute_Error", "mean"),
            Median_Error=("Absolute_Error", "median"),
            Mean_Percentage_Error=(
                "Percentage_Error",
                "mean"
            )
        )
        .reset_index()
    )

    fig = px.bar(
        risk_summary,
        x="Risk Category",
        y="MAE",
        title="Validation Error by Energy-Risk Category",
        labels={
            "MAE": "Mean Absolute Error (kWh)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        risk_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# BENCHMARKS & VALIDATION
# ============================================================

elif page == "Benchmarks & Validation":

    st.title("Benchmarks & Validation Analytics")

    st.subheader("Model Benchmark")

    st.dataframe(
        benchmark_results.style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            benchmark_results,
            x="Model",
            y="RMSE",
            title="Model Comparison: RMSE",
            labels={
                "RMSE": "RMSE (kWh)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            benchmark_results,
            x="Model",
            y="R²",
            title="Model Comparison: R²",
            labels={
                "R²": "R²"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Five-Fold Cross-Validation")

    st.dataframe(
        fold_df.style.format({
            "MAE": "{:.6f}",
            "RMSE": "{:.6f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = px.box(
        pd.DataFrame({
            "Fold": fold_df["Fold"].astype(str),
            "R²": fold_df["R²"]
        }),
        x="Fold",
        y="R²",
        title="Cross-Validation R² Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "OOF MAE",
            f"{oof_mae:.4f}"
        )

    with c2:
        metric_card(
            "OOF RMSE",
            f"{oof_rmse:.4f}"
        )

    with c3:
        metric_card(
            "OOF R²",
            f"{oof_r2:.4f}"
        )

    st.subheader("Actual vs Predicted")

    fig = px.scatter(
        validation_df,
        x="Actual",
        y="Predicted",
        title="Out-of-Fold Actual vs Predicted Energy"
    )

    min_value = min(
        validation_df["Actual"].min(),
        validation_df["Predicted"].min()
    )

    max_value = max(
        validation_df["Actual"].max(),
        validation_df["Predicted"].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect Prediction"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Residual Analysis")

    fig = px.histogram(
        validation_df,
        x="Error",
        nbins=40,
        title="Out-of-Fold Prediction Error Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.title("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Gradient Boosting Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Interpretation")

    dominant_feature = (
        importance_df.iloc[0]["Feature"]
    )

    st.write(
        f"The model identifies **{dominant_feature}** as the most "
        "important predictor of AI workload energy consumption."
    )

    st.write(
        "Feature importance indicates predictive contribution within "
        "this model. It should not be interpreted as proof of a causal relationship."
    )


# ============================================================
# ENERGY ANALYSIS
# ============================================================

elif page == "Energy Analysis":

    st.title("Energy Consumption Analysis")

    st.subheader("Energy vs Request Rate")

    energy_rate = (
        df.groupby(
            [
                "request_rate_x",
                "hf-output-len"
            ]
        )["energy_kWh"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        energy_rate,
        x="request_rate_x",
        y="energy_kWh",
        color="hf-output-len",
        markers=True,
        title="Average Energy Consumption vs Request Rate",
        labels={
            "request_rate_x": "Request Rate",
            "energy_kWh": "Average Energy (kWh)",
            "hf-output-len": "Output Length"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Energy vs Output Length")

    fig = px.box(
        df,
        x="hf-output-len",
        y="energy_kWh",
        title="Energy Consumption by Output Length",
        labels={
            "hf-output-len": "Output Length",
            "energy_kWh": "Energy (kWh)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Energy Distribution")

    fig = px.histogram(
        df,
        x="energy_kWh",
        nbins=40,
        title="Distribution of Energy Consumption"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# CARBON ANALYSIS
# ============================================================

elif page == "Carbon Analysis":

    st.title("Carbon Analysis")

    st.write(
        "Estimated carbon emissions are calculated by combining "
        "predicted energy consumption with electricity-grid carbon intensity."
    )

    request_rate = st.slider(
        "Request Rate",
        10,
        1000,
        100,
        10
    )

    output_length = st.selectbox(
        "Output Length",
        sorted(
            df["hf-output-len"].unique()
        )
    )

    country = st.selectbox(
        "Electricity System",
        list(carbon_intensity.keys())
    )

    input_df = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    predicted_energy = model.predict(
        input_df
    )[0]

    intensity = carbon_intensity[country]

    carbon = (
        predicted_energy
        * intensity
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Predicted Energy",
            f"{predicted_energy:.4f} kWh"
        )

    with c2:
        metric_card(
            "Grid Carbon Intensity",
            f"{intensity:.2f} gCO₂/kWh"
        )

    with c3:
        metric_card(
            "Estimated Carbon",
            f"{carbon:.2f} gCO₂"
        )

    st.subheader("Carbon Formula")

    st.latex(
        r"""
        \text{Carbon Emissions}
        =
        \text{Energy Consumption}
        \times
        \text{Grid Carbon Intensity}
        """
    )


# ============================================================
# COUNTRY COMPARISON
# ============================================================

elif page == "Country Comparison":

    st.title("Country Comparison")

    request_rate = st.slider(
        "Request Rate",
        10,
        1000,
        100,
        10
    )

    output_length = st.selectbox(
        "Output Length",
        sorted(
            df["hf-output-len"].unique()
        )
    )

    input_df = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    energy = model.predict(
        input_df
    )[0]

    comparison = pd.DataFrame({
        "Country": list(
            carbon_intensity.keys()
        ),
        "Carbon Intensity": list(
            carbon_intensity.values()
        )
    })

    comparison["Estimated Carbon"] = (
        energy
        * comparison["Carbon Intensity"]
    )

    comparison = comparison.sort_values(
        "Estimated Carbon"
    )

    fig = px.bar(
        comparison,
        x="Country",
        y="Estimated Carbon",
        title="Estimated Carbon Footprint by Electricity System",
        labels={
            "Estimated Carbon": "Estimated CO₂ (g)",
            "Country": ""
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        comparison.style.format({
            "Carbon Intensity": "{:.2f}",
            "Estimated Carbon": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This is a scenario comparison. It does not mean that the ML model "
        "itself reduces emissions. The comparison reflects differences in "
        "electricity-grid carbon intensity."
    )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

elif page == "What-If Analysis":

    st.title("What-If Analysis")

    st.write(
        "Explore how workload characteristics influence predicted "
        "energy consumption and estimated carbon emissions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        request_rate = st.slider(
            "Request Rate",
            10,
            1000,
            100,
            10
        )

    with col2:

        output_length = st.selectbox(
            "Output Length",
            sorted(
                df["hf-output-len"].unique()
            )
        )

    with col3:

        country = st.selectbox(
            "Country",
            list(
                carbon_intensity.keys()
            )
        )

    input_df = pd.DataFrame({
        "request_rate_x": [request_rate],
        "hf-output-len": [output_length]
    })

    energy = model.predict(
        input_df
    )[0]

    carbon = (
        energy
        * carbon_intensity[country]
    )

    c1, c2 = st.columns(2)

    with c1:
        metric_card(
            "Predicted Energy",
            f"{energy:.4f} kWh"
        )

    with c2:
        metric_card(
            "Estimated Carbon",
            f"{carbon:.2f} gCO₂"
        )

    st.subheader("Request Rate Scenario")

    rates = np.array([
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        200, 300, 400, 500,
        600, 700, 800, 900, 1000
    ])

    scenario = pd.DataFrame({
        "request_rate_x": rates,
        "hf-output-len": output_length
    })

    scenario["Predicted Energy"] = model.predict(
        scenario[
            [
                "request_rate_x",
                "hf-output-len"
            ]
        ]
    )

    scenario["Estimated Carbon"] = (
        scenario["Predicted Energy"]
        * carbon_intensity[country]
    )

    fig = px.line(
        scenario,
        x="request_rate_x",
        y="Predicted Energy",
        markers=True,
        title="Predicted Energy Across Request Rates"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.line(
        scenario,
        x="request_rate_x",
        y="Estimated Carbon",
        markers=True,
        title="Estimated Carbon Across Request Rates"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# RESEARCH FINDINGS
# ============================================================

elif page == "Research Findings":

    st.title("Research Findings")

    st.subheader("1. Request Rate is a Major Predictor")

    st.write(
        "Request rate is the strongest predictor identified by the "
        "Gradient Boosting model. The relationship is nonlinear, with "
        "energy consumption changing sharply at lower request rates and "
        "then approaching a plateau."
    )

    st.subheader("2. Output Length Influences Energy Consumption")

    st.write(
        "Longer requested outputs generally require more energy. "
        "The relationship is not perfectly proportional, indicating "
        "that workload behaviour is more complex than a simple linear rule."
    )

    st.subheader("3. Nonlinear Models Perform Better")

    st.write(
        "Gradient Boosting substantially outperformed the linear "
        "regression baselines and the evaluated neural network in the "
        "current experimental setting."
    )

    st.subheader("4. Carbon Depends on the Electricity System")

    st.write(
        "The same predicted AI energy consumption can correspond to "
        "very different carbon emissions depending on the carbon intensity "
        "of the electricity system supplying the workload."
    )

    st.subheader("5. The Model Should Be Used Within Its Scope")

    st.write(
        "The model was developed using a specific AI inference workload "
        "dataset. Its predictions should therefore not automatically be "
        "generalised to every AI model, hardware configuration, or data centre."
    )


# ============================================================
# METHODOLOGY
# ============================================================

elif page == "Methodology & Limitations":

    st.title("Methodology & Limitations")

    st.subheader("Machine Learning Workflow")

    st.markdown("""
    **1. Data Collection**

    AI inference workload measurements.

    **2. Data Cleaning**

    Incomplete runs were excluded.

    **3. Feature Selection**

    Request rate and output length were used as pre-run workload predictors.

    **4. Model Development**

    Gradient Boosting, Random Forest, Linear Regression, Ridge Regression,
    and a Neural Network were evaluated.

    **5. Validation**

    Five-fold GroupKFold validation was used, with workload-condition groups
    kept separate between training and validation.

    **6. Evaluation**

    MAE, RMSE and R² were used.

    **7. Interpretation**

    Feature importance and workload-response analysis were performed.

    **8. Carbon Extension**

    Predicted energy was combined with electricity-grid carbon intensity.
    """)

    st.subheader("Important Limitations")

    limitations = [
        "The dataset represents a specific AI inference workload rather than all AI workloads.",
        "The model uses request rate and output length as the main predictors.",
        "Carbon calculations depend on the selected electricity carbon-intensity data.",
        "Country comparisons are scenario analyses rather than a live carbon-aware scheduling system.",
        "The model should not be interpreted as establishing causal relationships.",
        "Predictions outside the observed workload range may be unreliable.",
        "Reliability indicators are based on validation error and are not probabilities."
    ]

    for item in limitations:
        st.write("• " + item)

    st.subheader("Why Grouped Validation?")

    st.write(
        "The dataset contains repeated measurements for combinations of "
        "request rate and output length. Grouped cross-validation reduces "
        "the risk of placing nearly identical experimental conditions in "
        "both training and validation sets."
    )
