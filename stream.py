import streamlit as st
import pickle
import joblib
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c63ff !important;
        color: white !important;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 28px;
        color: white;
        margin: 16px 0;
    }
    .result-box h2 { font-size: 2.4rem; margin: 0; }
    .result-box p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.95rem; }
    .future-box {
        background: #eef2ff;
        border-left: 4px solid #6c63ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
    }
    .good-invest {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 14px;
        padding: 24px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .bad-invest {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        border-radius: 14px;
        padding: 24px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e8eaf0;
        margin-bottom: 10px;
    }
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #e8eaf0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    reg_model       = pickle.load(open("regression_model.pkl", "rb"))
    clf_model       = joblib.load("classification.pkl")
    encoders        = pickle.load(open("label.pkl", "rb"))
    standard_scaler = pickle.load(open("standard_scaler.pkl", "rb"))
    minmax_scaler   = pickle.load(open("minmax_scaler.pkl", "rb"))
    return reg_model, clf_model, encoders, standard_scaler, minmax_scaler

reg_model, clf_model, encoders, standard_scaler, minmax_scaler = load_artifacts()


# ── Safe encode helper ────────────────────────────────────────────────────────
def safe_encode(enc, value):
    if value not in enc.classes_:
        st.error(f"⚠️ '{value}' not found in encoder. Available: {list(enc.classes_)}")
        st.stop()
    return int(enc.transform([value])[0])


# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 House Price & Investment Predictor")
st.markdown("---")

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Property Details")

    st.markdown("#### 📍 Location")
    state    = st.selectbox("State",    sorted(encoders["State"].classes_))
    city     = st.selectbox("City",     sorted(encoders["City"].classes_))
    locality = st.selectbox("Locality", sorted(encoders["Locality"].classes_))

    st.markdown("---")
    st.markdown("#### 🏗️ Property Info")
    property_type = st.selectbox("Property Type", encoders["Property_Type"].classes_)
    bhk           = st.selectbox("BHK", [1, 2, 3, 4, 5])

    # Size: training data shows range ~500–5000 sqft
    size = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1200, step=50)

    # ── CRITICAL FIX ──────────────────────────────────────────────────────────
    # Your training data Price_per_SqFt values are 0.010 to 0.325 (normalized).
    # We take rupees from the user and convert back to training scale.
    # Training range 0.010–0.325 corresponds roughly to ₹500–₹20000/sqft.
    # Formula: normalized = actual_price / max_price_in_training
    # Based on your value_counts the max normalized value is 0.325 →
    # we divide user input by a factor so it maps into [0.0, 0.325].
    # Adjust PRICE_SCALE to match your actual training normalization.
    PRICE_SCALE = 61538.0  # 0.325 * 61538 ≈ 20000 (max ₹/sqft assumed)

    price_per_sqft_display = st.number_input(
        "Price per sq ft (₹)",
        min_value=500, max_value=30000, value=5000, step=100,
        help="Enter actual rupees per sqft. Internally converted to model scale."
    )
    # Convert to model's normalized scale
    price_per_sqft = price_per_sqft_display / PRICE_SCALE

    furnished    = st.selectbox("Furnished Status", encoders["Furnished_Status"].classes_)
    floor_no     = st.number_input("Floor No",       min_value=0,  max_value=100, value=5,  step=1)
    total_floors = st.number_input("Total Floors",   min_value=1,  max_value=100, value=15, step=1)
    age          = st.number_input("Age of Property (years)", min_value=0, max_value=100, value=3, step=1)
    facing       = st.selectbox("Facing", encoders["Facing"].classes_)

    st.markdown("---")
    st.markdown("#### 🏥 Surroundings")
    schools   = st.number_input("Nearby Schools",   min_value=0, max_value=20, value=3, step=1)
    hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=2, step=1)
    transport = st.selectbox("Public Transport Accessibility",
                             encoders["Public_Transport_Accessibility"].classes_)
    parking   = st.selectbox("Parking Space",  encoders["Parking_Space"].classes_)
    security  = st.selectbox("Security",       encoders["Security"].classes_)
    amenities = st.selectbox("Amenities Score", sorted(encoders["Amenities"].classes_))
    owner_type = st.selectbox("Owner Type",    encoders["Owner_Type"].classes_)
    avail      = st.selectbox("Availability Status", encoders["Availability_Status"].classes_)


# ── Encode ────────────────────────────────────────────────────────────────────
state_enc     = safe_encode(encoders["State"],                          state)
city_enc      = safe_encode(encoders["City"],                           city)
locality_enc  = safe_encode(encoders["Locality"],                       locality)
prop_enc      = safe_encode(encoders["Property_Type"],                  property_type)
furn_enc      = safe_encode(encoders["Furnished_Status"],               furnished)
transport_enc = safe_encode(encoders["Public_Transport_Accessibility"], transport)
parking_enc   = safe_encode(encoders["Parking_Space"],                  parking)
security_enc  = safe_encode(encoders["Security"],                       security)
amenities_enc = safe_encode(encoders["Amenities"],                      amenities)
facing_enc    = safe_encode(encoders["Facing"],                         facing)
owner_enc     = safe_encode(encoders["Owner_Type"],                     owner_type)
avail_enc     = safe_encode(encoders["Availability_Status"],            avail)


# ── Feature vector (order must match training exactly) ────────────────────────
features = np.array([[
    state_enc, city_enc, locality_enc, prop_enc,
    int(bhk), float(size), float(price_per_sqft),   # price_per_sqft is now normalized
    furn_enc, float(floor_no), float(total_floors), float(age),
    float(schools), float(hospitals),
    transport_enc, parking_enc, security_enc,
    amenities_enc, facing_enc, owner_enc, avail_enc
]])

COL_NAMES = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_per_SqFt (normalized)", "Furnished_Status", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Amenities", "Facing", "Owner_Type", "Availability_Status"
]

# ── Debug expander (always visible so you can verify) ─────────────────────────
with st.expander("🔍 Debug panel — verify feature vector before predicting"):
    st.markdown("**Raw feature vector sent to model:**")
    import pandas as pd
    debug_df = pd.DataFrame([features[0]], columns=COL_NAMES)
    st.dataframe(debug_df, use_container_width=True)
    st.caption(f"Price_per_SqFt normalized = {price_per_sqft:.4f}  "
               f"(₹{price_per_sqft_display:,} ÷ {PRICE_SCALE:,.0f})")
    st.caption("If this value is outside [0.010, 0.325], adjust PRICE_SCALE in the code.")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💰 Price Prediction", "📊 Investment Analysis"])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — REGRESSION
# ════════════════════════════════════════════════════════════════════
with tab1:

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown("##### Predict the market price for the selected property")
    with col_btn:
        predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        features_scaled = standard_scaler.transform(features)

        # Sanity check — show scaled values
        with st.expander("🔬 StandardScaler output (should vary when you change inputs)"):
            scaled_df = pd.DataFrame([features_scaled[0]], columns=COL_NAMES)
            st.dataframe(scaled_df, use_container_width=True)
            st.caption("If all rows look the same across different inputs, "
                       "your scaler was fit on different data than what you're passing.")

        price = float(reg_model.predict(features_scaled)[0])

        # ── Result card ──
        st.markdown(f"""
        <div class="result-box">
            <p>Predicted Market Price</p>
            <h2>₹ {price:.2f} Lakhs</h2>
            <p>For a {bhk} BHK · {size} sq ft · {property_type} in {locality}, {city}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics row ──
        c1, c2, c3 = st.columns(3)
        c1.metric("Price per Sq Ft (input)", f"₹ {price_per_sqft_display:,}")
        c2.metric("Normalized (model input)", f"{price_per_sqft:.4f}")
        c3.metric("Total Area", f"{size:,} sq ft")

        # ── 5-year projection ──
        st.markdown("---")
        st.markdown("##### 📈 5-Year Price Projection")
        growth_rate = st.slider("Annual Growth Rate (%)", 0, 15, 5, key="growth") / 100
        future      = price * ((1 + growth_rate) ** 5)
        gain        = future - price

        st.markdown(f"""
        <div class="future-box">
            <strong>Projected price after 5 years @ {growth_rate*100:.0f}% annual growth</strong><br>
            <span style="font-size:1.5rem;color:#6c63ff;font-weight:700;">₹ {future:.2f} Lakhs</span>
            &nbsp;&nbsp;
            <span style="color:#27ae60;font-weight:600;">(+₹ {gain:.2f} Lakhs gain)</span>
        </div>
        """, unsafe_allow_html=True)

        # Year-by-year table
        years_data = {
            "Year": list(range(1, 6)),
            "Projected Price (Lakhs)": [
                round(price * ((1 + growth_rate) ** y), 2) for y in range(1, 6)
            ],
            "Gain over today (Lakhs)": [
                round(price * ((1 + growth_rate) ** y) - price, 2) for y in range(1, 6)
            ]
        }
        st.dataframe(pd.DataFrame(years_data), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════
with tab2:

    col_info2, col_btn2 = st.columns([3, 1])
    with col_info2:
        st.markdown("##### Classify if this property is a good real-estate investment")
    with col_btn2:
        invest_btn = st.button("📊 Evaluate", type="primary", use_container_width=True)

    if invest_btn:
        features_scaled_mm = minmax_scaler.transform(features)

        with st.expander("🔬 MinMaxScaler output"):
            scaled_df2 = pd.DataFrame([features_scaled_mm[0]], columns=COL_NAMES)
            st.dataframe(scaled_df2, use_container_width=True)

        pred = clf_model.predict(features_scaled_mm)[0]
        prob = clf_model.predict_proba(features_scaled_mm)[0] \
               if hasattr(clf_model, "predict_proba") else None

        if pred == 1:
            st.markdown("""
            <div class="good-invest">
                ✅ &nbsp; Good Investment
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="bad-invest">
                ❌ &nbsp; Not a Good Investment
            </div>
            """, unsafe_allow_html=True)

        if prob is not None:
            st.markdown("---")
            good_conf = float(prob[1]) * 100
            bad_conf  = float(prob[0]) * 100

            c1, c2 = st.columns(2)
            c1.metric("✅ Good Investment Confidence", f"{good_conf:.1f}%")
            c2.metric("❌ Not Good Confidence",        f"{bad_conf:.1f}%")

            st.progress(float(prob[1]), text=f"Investment quality score: {good_conf:.1f}%")

            # Interpretation
            if good_conf >= 70:
                st.success("🟢 High confidence — strong investment signal")
            elif good_conf >= 50:
                st.warning("🟡 Moderate confidence — consider further research")
            else:
                st.error("🔴 Low confidence — this property may not be a strong buy")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999;font-size:0.8rem;'>"
    "🏠 House Price Predictor · Powered by ML · "
    "Adjust <code>PRICE_SCALE</code> if predictions still seem flat"
    "</p>",
    unsafe_allow_html=True
)