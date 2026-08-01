"""
=============================================================================
 STEP 4 : MODEL DEPLOYMENT  (tourism_project/deployment/app.py)
=============================================================================
Streamlit front end for the Wellness Tourism Package propensity model.

How it fits the pipeline
------------------------
The GitHub Actions training job saves `best_tourism_model_v1.joblib` into this
same folder and commits it back to `main`. Streamlit Community Cloud watches
the repository, so every successful pipeline run automatically refreshes the
model that this app serves - no manual redeploy required.

Because the saved artifact is a full scikit-learn Pipeline (imputation +
scaling + one-hot encoding + XGBoost), this file only has to collect raw,
human-readable inputs into a one-row DataFrame with the original column names.
All the preprocessing happens inside the loaded model.
=============================================================================
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="🧳",
    layout="wide",
)

# The model sits next to this script, so resolve the path from __file__ rather
# than the working directory Streamlit Cloud happens to launch from.
MODEL_PATH = Path(__file__).resolve().parent / "best_tourism_model_v1.joblib"

# The model expects these columns, in this order.
FEATURE_ORDER = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


@st.cache_resource
def load_model():
    """Load the committed pipeline once and keep it in memory across reruns."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🧳 Wellness Tourism Package - Purchase Prediction")
st.markdown(
    """
**Visit with Us** wants to pitch its new *Wellness Tourism Package* only to the
customers most likely to buy, instead of calling the whole database.

Enter a customer's profile below and the model returns the probability that
this customer will purchase the package, so the sales team can prioritise
its follow-ups.
"""
)

if model is None:
    st.error(
        "Model file `best_tourism_model_v1.joblib` was not found in the deployment "
        "folder.\n\nRun the GitHub Actions pipeline first - the training job trains "
        "the model and commits it into `tourism_project/deployment/`."
    )
    st.stop()

st.success("Model loaded successfully from the repository.")
st.divider()

# -----------------------------------------------------------------------------
# Input widgets, grouped the same way the data dictionary groups the fields.
# -----------------------------------------------------------------------------
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

with c2:
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income (gross)", min_value=1000, max_value=200000, value=22400, step=500
    )
    city_tier = st.selectbox(
        "City Tier", [1, 2, 3], index=0, help="Tier 1 is the most developed."
    )
    passport = st.selectbox(
        "Holds a valid Passport?", [0, 1], index=0,
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

with c3:
    own_car = st.selectbox(
        "Owns a Car?", [0, 1], index=1,
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    number_of_trips = st.number_input(
        "Average Trips per Year", min_value=0, max_value=30, value=3, step=1
    )
    persons_visiting = st.number_input(
        "Total People Visiting", min_value=1, max_value=10, value=3, step=1
    )
    children_visiting = st.number_input(
        "Children (under 5) Visiting", min_value=0, max_value=5, value=1, step=1
    )

st.subheader("Sales interaction details")
c4, c5, c6 = st.columns(3)

with c4:
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )

with c5:
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)", min_value=1, max_value=180, value=14, step=1
    )
    number_of_followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=4, step=1
    )

with c6:
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5], index=0)

# -----------------------------------------------------------------------------
# Assemble the inputs into a single-row DataFrame with the training column names.
# -----------------------------------------------------------------------------
input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }
    ]
)[FEATURE_ORDER]

with st.expander("Review the record sent to the model"):
    st.dataframe(input_data, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Prediction. The decision threshold is exposed because the business cost of a
# missed buyer (lost revenue) differs from a wasted call (a few minutes of
# agent time) - marketing can tune it without retraining.
# -----------------------------------------------------------------------------
threshold = st.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Lower it to contact more customers (higher recall, more wasted calls).",
)

if st.button("Predict Purchase Likelihood", type="primary", use_container_width=True):
    probability = float(model.predict_proba(input_data)[0][1])
    prediction = int(probability >= threshold)

    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Purchase probability", f"{probability:.1%}")
    with r2:
        if prediction == 1:
            st.success(
                f"**LIKELY TO PURCHASE** - probability {probability:.1%} is at or above "
                f"the {threshold:.0%} threshold. Prioritise this customer for the "
                "Wellness Tourism Package pitch."
            )
        else:
            st.warning(
                f"**UNLIKELY TO PURCHASE** - probability {probability:.1%} is below the "
                f"{threshold:.0%} threshold. Deprioritise, or nurture with a lower-cost "
                "channel such as email."
            )

    st.progress(min(probability, 1.0))
    st.caption(
        "The model scores propensity only. Treat it as a prioritisation aid for the "
        "sales team, not as an automatic decision."
    )

st.divider()
st.caption(
    "Visit with Us · Wellness Tourism Package · model trained and committed "
    "automatically by the GitHub Actions MLOps pipeline."
)
