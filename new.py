import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate  AI Analyser",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background-color:#080C14 !important; color:#E2E8F0 !important; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:1.5rem 2rem 3rem 2rem !important; max-width:1300px; }
::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-track{background:#0D1117} ::-webkit-scrollbar-thumb{background:#2D3748;border-radius:3px}

.hero-wrap { position:relative; border-radius:24px; overflow:hidden; margin-bottom:1.8rem; background:linear-gradient(135deg,#0D1B2A,#112240,#0A192F); border:1px solid rgba(99,179,237,0.12); padding:42px 52px; }
.hero-wrap::before { content:''; position:absolute; inset:0; background:radial-gradient(ellipse 60% 80% at 80% 50%,rgba(56,189,248,0.08),transparent 70%),radial-gradient(ellipse 40% 60% at 15% 30%,rgba(139,92,246,0.07),transparent 60%); pointer-events:none; }
.hero-inner { display:flex; justify-content:space-between; align-items:center; gap:2rem; position:relative; }
.hero-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.25); color:#38BDF8; font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; padding:5px 14px; border-radius:100px; margin-bottom:14px; }
.hero-title { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; line-height:1.1; color:#F7FAFC; margin:0 0 10px 0; }
.hero-title span { background:linear-gradient(90deg,#38BDF8,#818CF8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { color:#94A3B8; font-size:0.95rem; font-weight:300; line-height:1.65; max-width:460px; margin:0; }
.hero-stats-row { display:flex; gap:10px; flex-wrap:wrap; }
.stat-box { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:18px 24px; text-align:center; min-width:90px; }
.stat-num { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; color:#38BDF8; line-height:1; }
.stat-lbl { font-size:0.68rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }

[data-testid="stSidebar"] { background:#0B1120 !important; border-right:1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
.sb-logo { font-family:'Syne',sans-serif; font-weight:800; font-size:1.25rem; color:#F7FAFC; padding:0 0.5rem 1.2rem; }
.sb-logo span { color:#38BDF8; }
.sb-sec { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:14px; margin-bottom:12px; }
.sb-sec-title { font-family:'Syne',sans-serif; font-size:0.67rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#475569; margin-bottom:10px; padding-bottom:7px; border-bottom:1px solid rgba(255,255,255,0.05); }
[data-testid="stSidebar"] label,.stSelectbox label,.stNumberInput label,.stSlider label { color:#94A3B8 !important; font-size:0.78rem !important; font-weight:500 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important; border-radius:9px !important; color:#E2E8F0 !important; }
[data-baseweb="select"] > div:focus-within { border-color:#38BDF8 !important; box-shadow:0 0 0 2px rgba(56,189,248,0.15) !important; }
[data-testid="stNumberInput"] input { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important; border-radius:9px !important; color:#E2E8F0 !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { background:rgba(255,255,255,0.03) !important; border-radius:13px !important; padding:4px !important; border:1px solid rgba(255,255,255,0.07) !important; gap:3px !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; border-radius:9px !important; color:#64748B !important; font-family:'DM Sans',sans-serif !important; font-weight:500 !important; font-size:0.88rem !important; padding:9px 22px !important; border:none !important; }
[data-testid="stTabs"] [aria-selected="true"] { background:linear-gradient(135deg,#0EA5E9,#6366F1) !important; color:#fff !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"],[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }

.stButton > button { background:linear-gradient(135deg,#0EA5E9,#6366F1) !important; color:#fff !important; border:none !important; border-radius:12px !important; font-family:'DM Sans',sans-serif !important; font-weight:600 !important; font-size:0.93rem !important; padding:13px 28px !important; width:100% !important; transition:all .2s ease !important; box-shadow:0 4px 18px rgba(14,165,233,0.28) !important; }
.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(14,165,233,0.42) !important; }

[data-testid="stMetric"] { background:rgba(255,255,255,0.03) !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:13px !important; padding:18px !important; }
[data-testid="stMetricLabel"] { color:#64748B !important; font-size:0.72rem !important; text-transform:uppercase !important; letter-spacing:0.08em !important; font-weight:600 !important; }
[data-testid="stMetricValue"] { font-family:'Syne',sans-serif !important; color:#F7FAFC !important; font-size:1.5rem !important; }

.result-card { position:relative; border-radius:20px; padding:38px 40px; text-align:center; margin:20px 0; }
.rc-price { background:linear-gradient(135deg,#0C1F3F,#112240); border:1px solid rgba(56,189,248,0.2); box-shadow:0 0 60px rgba(56,189,248,0.07), inset 0 1px 0 rgba(255,255,255,0.04); }
.rc-label { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.14em; color:#38BDF8; font-weight:600; margin-bottom:10px; }
.rc-value { font-family:'Syne',sans-serif; font-size:3.6rem; font-weight:800; line-height:1; background:linear-gradient(90deg,#E0F2FE,#BAE6FD,#7DD3FC); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; }
.rc-sub { color:#64748B; font-size:0.83rem; }
.rc-good { background:linear-gradient(135deg,#052E16,#064E3B); border:1px solid rgba(34,197,94,0.25); box-shadow:0 0 55px rgba(34,197,94,0.07); }
.rc-bad  { background:linear-gradient(135deg,#2D0A0A,#450A0A); border:1px solid rgba(239,68,68,0.25); box-shadow:0 0 55px rgba(239,68,68,0.07); }
.rv-emoji { font-size:2.8rem; margin-bottom:10px; display:block; }
.rv-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin-bottom:8px; }
.rv-good { color:#4ADE80; } .rv-bad { color:#F87171; }
.rv-sub { color:#94A3B8; font-size:0.85rem; }

.conf-wrap { margin-top:22px; }
.conf-label { display:flex; justify-content:space-between; font-size:0.75rem; color:#64748B; margin-bottom:7px; letter-spacing:0.05em; text-transform:uppercase; }
.conf-bg { height:8px; background:rgba(255,255,255,0.06); border-radius:100px; overflow:hidden; }
.conf-fill { height:100%; border-radius:100px; }
.cf-good { background:linear-gradient(90deg,#22C55E,#4ADE80); }
.cf-bad  { background:linear-gradient(90deg,#EF4444,#F87171); }

.chips { display:flex; gap:10px; margin:18px 0; flex-wrap:wrap; }
.chip  { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:9px; padding:9px 16px; display:flex; align-items:center; gap:7px; font-size:0.8rem; color:#CBD5E1; }

.sec-head { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#E2E8F0; margin:26px 0 10px; display:flex; align-items:center; gap:8px; }
.sec-head::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.07); margin-left:8px; }

.footer { text-align:center; padding:28px 0 6px; color:#334155; font-size:0.76rem; letter-spacing:0.04em; border-top:1px solid rgba(255,255,255,0.05); margin-top:36px; }
.footer strong { color:#475569; }
.stAlert { border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# LABEL MAPS  (alphabetical = sklearn LabelEncoder default)
# ─────────────────────────────────────────────────────────
STATES = sorted([
    'Andhra Pradesh','Assam','Bihar','Chhattisgarh','Delhi','Gujarat',
    'Haryana','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra',
    'Odisha','Punjab','Rajasthan','Tamil Nadu','Telangana','Uttar Pradesh',
    'Uttarakhand','West Bengal'
])
CITIES = sorted([
    'Ahmedabad','Amritsar','Bangalore','Bhopal','Bhubaneswar','Bilaspur',
    'Chennai','Coimbatore','Cuttack','Dehradun','Durgapur','Dwarka',
    'Faridabad','Gaya','Gurgaon','Guwahati','Haridwar','Hyderabad',
    'Indore','Jaipur','Jamshedpur','Jodhpur','Kochi','Kolkata','Lucknow',
    'Ludhiana','Mangalore','Mumbai','Mysore','Nagpur','New Delhi','Noida',
    'Patna','Pune','Raipur','Ranchi','Silchar','Surat','Trivandrum',
    'Vijayawada','Vishakhapatnam','Warangal'
])
PROPERTY_TYPES = sorted(['Apartment','Independent House','Villa'])
FURNISHED      = sorted(['Furnished','Semi-furnished','Unfurnished'])
TRANSPORT      = sorted(['High','Low','Medium'])
PARKING        = sorted(['No','Yes'])
SECURITY       = sorted(['No','Yes'])
FACING         = sorted(['East','North','South','West'])
OWNER_TYPE     = sorted(['Broker','Builder','Owner'])
AVAILABILITY   = sorted(['Ready_to_Move','Under_Construction'])

def encode(lst, value):
    """Integer index in sorted list = LabelEncoder output."""
    try:
        return int(lst.index(value))
    except ValueError:
        st.error(f"❌ '{value}' not found. Available: {lst}")
        st.stop()

CITY_TIERS = {
    "Mumbai","New Delhi","Bangalore","Hyderabad","Chennai",
    "Kolkata","Pune","Ahmedabad","Gurgaon","Noida",
}

CITY_STATE = {
    'Hyderabad':'Telangana','Warangal':'Telangana',
    'Mumbai':'Maharashtra','Pune':'Maharashtra','Nagpur':'Maharashtra',
    'Chennai':'Tamil Nadu','Coimbatore':'Tamil Nadu',
    'Bangalore':'Karnataka','Mangalore':'Karnataka','Mysore':'Karnataka',
    'Kolkata':'West Bengal','Durgapur':'West Bengal',
    'Ahmedabad':'Gujarat','Surat':'Gujarat',
    'New Delhi':'Delhi','Dwarka':'Delhi',
    'Faridabad':'Haryana','Gurgaon':'Haryana',
    'Noida':'Uttar Pradesh','Lucknow':'Uttar Pradesh',
    'Jaipur':'Rajasthan','Jodhpur':'Rajasthan',
    'Bhopal':'Madhya Pradesh','Indore':'Madhya Pradesh',
    'Bhubaneswar':'Odisha','Cuttack':'Odisha',
    'Patna':'Bihar','Gaya':'Bihar',
    'Ranchi':'Jharkhand','Jamshedpur':'Jharkhand',
    'Guwahati':'Assam','Silchar':'Assam',
    'Kochi':'Kerala','Trivandrum':'Kerala',
    'Amritsar':'Punjab','Ludhiana':'Punjab',
    'Haridwar':'Uttarakhand','Dehradun':'Uttarakhand',
    'Vishakhapatnam':'Andhra Pradesh','Vijayawada':'Andhra Pradesh',
    'Raipur':'Chhattisgarh','Bilaspur':'Chhattisgarh',
}


# ─────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg  = pickle.load(open("regression_model.pkl","rb"))
            clf  = joblib.load("classification.pkl")
            ss   = pickle.load(open("standard_scaler.pkl","rb"))
            mm   = pickle.load(open("minmax_scaler.pkl","rb"))
            return reg, clf, ss, mm
        except Exception as e:
            st.error(f"❌ Model load error: {e}")
            st.stop()

reg_model, clf_model, std_scaler, mm_scaler = load_models()


# ─────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-inner">
    <div>
      <div class="hero-badge">🏙️ &nbsp; AI-Powered Real Estate</div>
      <h1 class="hero-title">REAL ESTATE HOUSE PRICE PREDICTOR <span>AI</span></h1>
      <p class="hero-sub">Predict property prices and evaluate investment potential across 42 Indian cities — powered by trained ML models.</p>
    </div>
    <div class="hero-stats-row">
      <div class="stat-box"><div class="stat-num">42</div><div class="stat-lbl">Cities</div></div>
      <div class="stat-box"><div class="stat-num">500</div><div class="stat-lbl">Localities</div></div>
      <div class="stat-box"><div class="stat-num">20</div><div class="stat-lbl">Features</div></div>
      <div class="stat-box"><div class="stat-num">250K</div><div class="stat-lbl">Trained On</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">🏙️ Prop<span>Sense</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-title">📍 Location</div>', unsafe_allow_html=True)
    city     = st.selectbox("City", CITIES, index=CITIES.index("Hyderabad"))
    locality = st.selectbox("Locality", [f"Locality_{i}" for i in range(1, 501)])
    state    = CITY_STATE.get(city, STATES[0])
    st.caption(f"📌 State: **{state}**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-title">🏠 Property</div>', unsafe_allow_html=True)
    property_type = st.selectbox("Type",       ["Apartment","Villa","Independent House"])
    bhk           = st.slider("BHK",            1, 5, 2)
    size          = st.number_input("Size (sq ft)", 100, 10000, 1200, step=50)
    furnished     = st.selectbox("Furnishing",  ["Unfurnished","Semi-furnished","Furnished"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-title">🏗️ Building</div>', unsafe_allow_html=True)
    floor      = st.number_input("Floor No.",     0, 30, 5)
    total_fl   = st.number_input("Total Floors",  1, 30, 10)
    year_built = st.number_input("Year Built", 1990, 2024, 2018)
    age        = 2025 - year_built
    st.caption(f"Property Age: **{age} yrs**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-title">💰 Pricing</div>', unsafe_allow_html=True)
    price_raw = st.slider("Price per sq ft (₹)", 500, 30000, 5000, step=100)
    # Map ₹500–30000 → dataset range 0.01–0.99
    price_enc = round((price_raw - 500) / (30000 - 500) * 0.98 + 0.01, 2)
    price_enc = max(0.01, min(0.99, price_enc))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-title">🏘️ Neighbourhood</div>', unsafe_allow_html=True)
    nearby_schools   = st.slider("Nearby Schools",   1, 10, 5)
    nearby_hospitals = st.slider("Nearby Hospitals", 1, 10, 4)
    transport        = st.selectbox("Transport Access", ["High","Medium","Low"])
    facing           = st.selectbox("Facing",           ["East","West","North","South"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-sec-title">✨ Amenities & Ownership</div>', unsafe_allow_html=True)
    parking      = st.selectbox("Parking",      ["Yes","No"])
    security     = st.selectbox("Security",     ["Yes","No"])
    owner_type   = st.selectbox("Owner Type",   ["Owner","Broker","Builder"])
    availability = st.selectbox("Availability", ["Ready_to_Move","Under_Construction"])
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# FEATURE VECTOR — 20 features in exact training column order:
# State, City, Locality, Property_Type, BHK, Size_in_SqFt,
# Price_per_SqFt, Year_Built, Furnished_Status, Floor_No,
# Total_Floors, Age_of_Property, Nearby_Schools, Nearby_Hospitals,
# Public_Transport_Accessibility, Parking_Space, Security,
# Facing, Owner_Type, Availability_Status
# ─────────────────────────────────────────────────────────
loc_idx = int(locality.split("_")[1]) - 1   # Locality_1→0 … Locality_500→499

features = np.array([[
    encode(STATES,        state),
    encode(CITIES,        city),
    loc_idx,
    encode(PROPERTY_TYPES, property_type),
    int(bhk),
    float(size),
    float(price_enc),
    int(year_built),
    encode(FURNISHED,     furnished),
    int(floor),
    int(total_fl),
    int(age),
    int(nearby_schools),
    int(nearby_hospitals),
    encode(TRANSPORT,     transport),
    encode(PARKING,       parking),
    encode(SECURITY,      security),
    encode(FACING,        facing),
    encode(OWNER_TYPE,    owner_type),
    encode(AVAILABILITY,  availability),
]], dtype=float)


# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📊 Investment Analysis", "📋 Property Summary"])


# ══════════════════════════════════════════════════════════
# TAB 1 — PRICE PREDICTION
# ══════════════════════════════════════════════════════════
with tab1:
    c_btn, c_hint = st.columns([1,2])
    with c_btn:
        predict_clicked = st.button("🔮 Predict Price Now")
    with c_hint:
        st.markdown("<p style='color:#475569;font-size:0.8rem;padding-top:14px;'>Set property details in the sidebar, then click predict.</p>", unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner("Running regression model…"):
            time.sleep(0.45)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled = std_scaler.transform(features)
                price  = float(reg_model.predict(scaled)[0])

        st.balloons()

        st.markdown(f"""
        <div class="result-card rc-price">
            <div class="rc-label">Estimated Market Value</div>
            <div class="rc-value">₹ {price:,.2f} L</div>
            <div class="rc-sub">{bhk} BHK &nbsp;·&nbsp; {size:,} sq ft &nbsp;·&nbsp; {property_type} &nbsp;·&nbsp; {city}</div>
        </div>""", unsafe_allow_html=True)

        price_cr       = price / 100
        price_sqft_est = (price * 1e5) / size if size else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted (Lakhs)",  f"₹ {price:,.2f}")
        c2.metric("In Crores",          f"₹ {price_cr:,.3f} Cr")
        c3.metric("Implied ₹/sq ft",    f"₹ {price_sqft_est:,.0f}")
        c4.metric("Property Age",       f"{age} yrs")

        furn_ico = {"Furnished":"🛋️","Semi-furnished":"🪑","Unfurnished":"📦"}
        tier_label = "Tier 1" if city in CITY_TIERS else "Tier 2"
        st.markdown(f"""
        <div class="chips">
          <div class="chip">🏙️ {city} ({tier_label})</div>
          <div class="chip">{furn_ico[furnished]} {furnished}</div>
          <div class="chip">{'🅿️' if parking=='Yes' else '🚫'} Parking {parking}</div>
          <div class="chip">{'🔒' if security=='Yes' else '🔓'} Security {security}</div>
          <div class="chip">🚌 {transport} Transport</div>
          <div class="chip">🏢 Floor {floor}/{total_fl}</div>
          <div class="chip">🧭 {facing}-Facing</div>
          <div class="chip">👤 {owner_type}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">📈 Price Growth Projection</div>', unsafe_allow_html=True)
        g_pct = st.slider("Annual Growth Rate (%)", 1, 20, 7, key="g_sl")
        g = g_pct / 100
        yrs  = list(range(0, 11))
        vals = [price * ((1+g)**y) for y in yrs]
        lbls = ["Now"] + [f"Y{y}" for y in yrs[1:]]
        st.line_chart(pd.DataFrame({"Value (₹ L)": vals}, index=lbls), height=280, color="#38BDF8")

        st.markdown('<div class="sec-head">🗓️ 5-Year Forecast</div>', unsafe_allow_html=True)
        df_t = pd.DataFrame({
            "Year":              [f"Year {y}" for y in range(1,6)],
            "Projected (₹ L)":  [f"₹ {price*((1+g)**y):,.2f}" for y in range(1,6)],
            "Total Growth":      [f"+{((1+g)**y-1)*100:.1f}%" for y in range(1,6)],
            "Annual Gain (₹ L)":[f"₹ {price*((1+g)**y)-price*((1+g)**(y-1)):,.2f}" for y in range(1,6)],
        })
        st.dataframe(df_t, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — INVESTMENT ANALYSIS
# ══════════════════════════════════════════════════════════
with tab2:
    c_btn2, c_hint2 = st.columns([1,2])
    with c_btn2:
        invest_clicked = st.button("📊 Evaluate Investment")
    with c_hint2:
        st.markdown("<p style='color:#475569;font-size:0.8rem;padding-top:14px;'>Runs a Random Forest classifier on all 20 property parameters.</p>", unsafe_allow_html=True)

    if invest_clicked:
        with st.spinner("Classifying investment potential…"):
            time.sleep(0.45)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaled2 = mm_scaler.transform(features)
                pred    = clf_model.predict(scaled2)[0]

        is_good = (int(pred) == 1)

        if is_good:
            st.markdown("""
            <div class="result-card rc-good">
                <span class="rv-emoji">✅</span>
                <div class="rv-title rv-good">Good Investment</div>
                <div class="rv-sub">Strong potential based on location, pricing, neighbourhood & amenity signals.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card rc-bad">
                <span class="rv-emoji">⚠️</span>
                <div class="rv-title rv-bad">Not Recommended</div>
                <div class="rv-sub">The model flags this as weaker. Consider adjusting price, location, or specs.</div>
            </div>""", unsafe_allow_html=True)

        if hasattr(clf_model, "predict_proba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob = float(clf_model.predict_proba(scaled2)[0][1])
            pct     = prob * 100
            bar_cls = "cf-good" if is_good else "cf-bad"
            lbl     = "Investment Confidence" if is_good else "Risk Level"
            st.markdown(f"""
            <div class="conf-wrap">
              <div class="conf-label"><span>{lbl}</span><span>{pct:.1f}%</span></div>
              <div class="conf-bg"><div class="conf-fill {bar_cls}" style="width:{pct}%;"></div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head">🔍 Factor Breakdown</div>', unsafe_allow_html=True)
        scores = {
            "Location":      85 if city in CITY_TIERS else 55,
            "Neighbourhood": min(100, (nearby_schools + nearby_hospitals) * 5),
            "Amenities":     (30 if parking=="Yes" else 0) +
                             (30 if security=="Yes" else 0) +
                             (25 if transport=="High" else 12 if transport=="Medium" else 0) +
                             (15 if furnished=="Furnished" else 7 if furnished=="Semi-furnished" else 0),
            "Condition":     max(0, 100 - age * 3),
            "Floor":         min(100, max(20, 100 - abs(floor-6)*4)),
            "Size":          min(100, int((size/3000)*100)),
        }
        df_s = pd.DataFrame({"Factor": list(scores.keys()), "Score /100": list(scores.values())})
        st.bar_chart(df_s.set_index("Factor"), height=240, color="#38BDF8")

        fa1, fa2, fa3 = st.columns(3)
        fa1.metric("Location",    f"{scores['Location']}/100")
        fa2.metric("Amenities",   f"{scores['Amenities']}/100")
        fa3.metric("Condition",   f"{scores['Condition']}/100")


# ══════════════════════════════════════════════════════════
# TAB 3 — PROPERTY SUMMARY
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">📋 Full Configuration (with encoded values)</div>', unsafe_allow_html=True)
    df_sum = pd.DataFrame({
        "Parameter": [
            "State","City","Locality","Property Type","BHK","Size",
            "Price/sq ft (₹)","Price/sq ft (encoded)","Year Built","Property Age",
            "Furnishing","Floor","Total Floors","Nearby Schools","Nearby Hospitals",
            "Transport","Parking","Security","Facing","Owner Type","Availability"
        ],
        "Value": [
            state, city, locality, property_type, f"{bhk} BHK", f"{size:,} sq ft",
            f"₹ {price_raw:,}", f"{price_enc:.2f}", str(year_built), f"{age} yrs",
            furnished, str(floor), str(total_fl), str(nearby_schools), str(nearby_hospitals),
            transport, parking, security, facing, owner_type, availability
        ],
        "Encoded": [
            encode(STATES,state), encode(CITIES,city), loc_idx,
            encode(PROPERTY_TYPES,property_type), bhk, size,
            "–", price_enc, year_built, age,
            encode(FURNISHED,furnished), floor, total_fl,
            nearby_schools, nearby_hospitals,
            encode(TRANSPORT,transport), encode(PARKING,parking),
            encode(SECURITY,security), encode(FACING,facing),
            encode(OWNER_TYPE,owner_type), encode(AVAILABILITY,availability)
        ]
    })
    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-head">💡 Investment Tips</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.info("🏙️ **Tier-1 cities** like Mumbai, Bangalore, Hyderabad show stronger appreciation.")
        st.info("🛋️ **Furnished units** command 15–25% higher rentals and resale values.")
    with t2:
        st.info("🔒 **Security + Parking** significantly boost the investment classifier score.")
        st.info("📐 **Floors 4–8** in mid-rise buildings typically hit the valuation sweet spot.")


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>PropSense AI</strong> &nbsp;·&nbsp; RandomForest Regression + Classification
    &nbsp;·&nbsp; 20 Features &nbsp;·&nbsp; 42 Cities &nbsp;·&nbsp; 250K Training Samples
</div>
""", unsafe_allow_html=True)