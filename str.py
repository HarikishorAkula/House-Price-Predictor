import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Property Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# PREMIUM UI CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.hero {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.hero h1 { font-size: 2.5rem; margin-bottom: 5px; }

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    border: none;
}

.result-box {
    background: linear-gradient(135deg, #4f46e5, #9333ea);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    color: white;
}

.good-invest {
    background: linear-gradient(135deg, #10b981, #22c55e);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 1.6rem;
}
.bad-invest {
    background: linear-gradient(135deg, #ef4444, #f97316);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 1.6rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏠 Smart Property Intelligence</h1>
    <p>AI-powered house price prediction & investment analysis</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        reg_model = pickle.load(open("regression_model.pkl", "rb"))
        clf_model = joblib.load("classification.pkl")
        encoders = pickle.load(open("label.pkl", "rb"))
        standard_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
        minmax_scaler = pickle.load(open("minmax_scaler.pkl", "rb"))
        return reg_model, clf_model, encoders, standard_scaler, minmax_scaler
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

reg_model, clf_model, encoders, standard_scaler, minmax_scaler = load_artifacts()

# ─────────────────────────────────────────────────────────
# ENCODER HELPER
# ─────────────────────────────────────────────────────────
def safe_encode(enc, value):
    if value not in enc.classes_:
        st.error(f"{value} not in encoder")
        st.stop()
    return enc.transform([value])[0]

# ─────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Property Details")

    state = st.selectbox("State", encoders["State"].classes_)
    city = st.selectbox("City", encoders["City"].classes_)
    locality = st.selectbox("Locality", encoders["Locality"].classes_)
    property_type = st.selectbox("Property Type", encoders["Property_Type"].classes_)

    bhk = st.slider("BHK", 1, 5, 2)
    size = st.number_input("Size (sq ft)", 100, 10000, 1200)

    price_display = st.number_input("Price per sq ft (₹)", 500, 30000, 5000)
    PRICE_SCALE = 61538.0
    price_per_sqft = price_display / PRICE_SCALE

    furnished = st.selectbox("Furnished", encoders["Furnished_Status"].classes_)
    floor = st.number_input("Floor", 0, 100, 5)
    total = st.number_input("Total Floors", 1, 100, 10)
    age = st.number_input("Age", 0, 100, 3)

    transport = st.selectbox("Transport", encoders["Public_Transport_Accessibility"].classes_)
    parking = st.selectbox("Parking", encoders["Parking_Space"].classes_)
    security = st.selectbox("Security", encoders["Security"].classes_)

# ─────────────────────────────────────────────────────────
# ENCODE INPUT
# ─────────────────────────────────────────────────────────
features = np.array([[
    safe_encode(encoders["State"], state),
    safe_encode(encoders["City"], city),
    safe_encode(encoders["Locality"], locality),
    safe_encode(encoders["Property_Type"], property_type),
    bhk, size, price_per_sqft,
    safe_encode(encoders["Furnished_Status"], furnished),
    floor, total, age,
    3, 2,  # dummy schools/hospitals
    safe_encode(encoders["Public_Transport_Accessibility"], transport),
    safe_encode(encoders["Parking_Space"], parking),
    safe_encode(encoders["Security"], security),
    5,  # amenities dummy
    1,  # facing dummy
    1,  # owner dummy
    1   # availability dummy
]])

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💰 Price Prediction", "📊 Investment"])

# ─────────────────────────────────────────────────────────
# PRICE PREDICTION
# ─────────────────────────────────────────────────────────
with tab1:
    if st.button("🔮 Predict Price"):
        with st.spinner("Analyzing property..."):
            scaled = standard_scaler.transform(features)
            price = reg_model.predict(scaled)[0]

        st.balloons()

        st.markdown(f"""
        <div class="result-box">
            <h2>₹ {price:.2f} Lakhs</h2>
            <p>{bhk} BHK • {size} sq ft • {city}</p>
        </div>
        """, unsafe_allow_html=True)

        # KPI
        c1, c2, c3 = st.columns(3)
        c1.metric("Area", f"{size} sq ft")
        c2.metric("₹/sqft", f"{price_display}")
        c3.metric("Property Age", f"{age} yrs")

        # Chart
        growth = st.slider("Growth %", 0, 15, 5) / 100
        years = list(range(1, 6))
        values = [price * ((1 + growth) ** y) for y in years]

        df = pd.DataFrame({"Year": years, "Price": values})
        st.line_chart(df.set_index("Year"))

# ─────────────────────────────────────────────────────────
# INVESTMENT
# ─────────────────────────────────────────────────────────
with tab2:
    if st.button("📊 Evaluate Investment"):
        scaled = minmax_scaler.transform(features)
        pred = clf_model.predict(scaled)[0]

        if pred == 1:
            st.markdown('<div class="good-invest">✅ Good Investment</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="bad-invest">❌ Not a Good Investment</div>', unsafe_allow_html=True)

        if hasattr(clf_model, "predict_proba"):
            prob = clf_model.predict_proba(scaled)[0][1]
            st.progress(prob)
            st.write(f"Confidence: {prob*100:.2f}%")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<center>Built with ❤️ using Streamlit</center>", unsafe_allow_html=True)