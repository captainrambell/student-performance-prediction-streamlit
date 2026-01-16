import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student GPA Predictor")
st.write(
    "This app uses a public student performance dataset with GPA on a 0.0–4.0 scale. "
    "It predicts a student's GPA based on weekly study time, absences, tutoring, "
    "parental support, and extracurricular activities."
)

st.caption(
    "Note: GPA is on a 0.0–4.0 scale, where 4.0 typically corresponds to an A / excellent performance."
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Student_performance_data.csv")
    return df

df = load_data()

st.subheader("Dataset preview")
st.dataframe(df.head())

# -----------------------------
# Select features and target
# -----------------------------
TARGET = "GPA"
features = ["StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport", "Extracurricular"]

data = df[features + [TARGET]].dropna().copy()

X = data[features]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# -----------------------------
# Model performance
# -----------------------------
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model performance")
col1, col2 = st.columns(2)
col1.metric("R² on test set", f"{r2:.3f}")
col2.metric("RMSE on test set", f"{rmse:.2f}")

# -----------------------------
# Sidebar: user input controls
# -----------------------------
st.sidebar.header("Student input features")

def slider_from_series(label, s, step=1.0):
    return st.sidebar.slider(
        label,
        min_value=float(s.min()),
        max_value=float(s.max()),
        value=float(s.median()),
        step=step,
    )

study_time = slider_from_series("Weekly study time (hours)", data["StudyTimeWeekly"], step=0.5)
absences = slider_from_series("Number of absences", data["Absences"], step=1.0)

tutoring = slider_from_series("Tutoring (0=no, 1=yes)", data["Tutoring"], step=1.0)
parental_support = slider_from_series("Parental support (0=low, 4=very high)", data["ParentalSupport"], step=1.0)
extracurricular = slider_from_series("Extracurricular (0=no, 1=yes)", data["Extracurricular"], step=1.0)

input_data = pd.DataFrame(
    {
        "StudyTimeWeekly": [study_time],
        "Absences": [absences],
        "Tutoring": [tutoring],
        "ParentalSupport": [parental_support],
        "Extracurricular": [extracurricular],
    }
)

st.sidebar.subheader("Current input values")
st.sidebar.table(
    input_data.rename(index={0: ""})
)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Predicted GPA (0.0–4.0)")

if st.button("Predict GPA"):
    pred_gpa = model.predict(input_data)[0]
    st.success(f"Estimated GPA: **{pred_gpa:.2f} / 4.00**")
else:
    st.info("Adjust the inputs on the left and click **Predict GPA** to see the prediction.")

# -----------------------------
# Simple visualizations
# -----------------------------
st.subheader("Distribution of GPA")
gpa_rounded = data["GPA"].round(2)

gpa_counts = (
    gpa_rounded
    .value_counts()
    .sort_index()
    .rename_axis("GPA")
    .reset_index(name="count")
)

st.bar_chart(gpa_counts.set_index("GPA"))
st.subheader("Study time vs GPA")
scatter_data = data[["StudyTimeWeekly", "GPA"]].rename(columns={"GPA": "GPA_4_scale"})
st.scatter_chart(scatter_data)
