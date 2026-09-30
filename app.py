import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f13 0%, #1a1a24 100%);
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.badge {
    display: inline-block;
    background: rgba(124,92,252,0.15);
    border: 1px solid rgba(124,92,252,0.4);
    color: #9b7eff;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 16px;
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #a89cff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    line-height: 1.1;
}

.subtitle {
    color: #8888a0;
    font-size: 16px;
    font-weight: 400;
    line-height: 1.6;
    margin-bottom: 2rem;
}

.result-positive {
    background: rgba(34,197,94,0.1);
    border: 1.5px solid rgba(34,197,94,0.3);
    border-radius: 14px;
    padding: 24px;
    margin-top: 16px;
    animation: fadeUp 0.4s ease;
}

.result-negative {
    background: rgba(239,68,68,0.1);
    border: 1.5px solid rgba(239,68,68,0.3);
    border-radius: 14px;
    padding: 24px;
    margin-top: 16px;
    animation: fadeUp 0.4s ease;
}

.result-title-pos {
    font-size: 28px;
    font-weight: 800;
    color: #4ade80;
    letter-spacing: -0.02em;
}

.result-title-neg {
    font-size: 28px;
    font-weight: 800;
    color: #f87171;
    letter-spacing: -0.02em;
}

.result-sub {
    color: #8888a0;
    font-size: 14px;
    margin-top: 4px;
}

.conf-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8888a0;
    margin-top: 16px;
    margin-bottom: 6px;
}

.stTextArea textarea {
    background: #22222f !important;
    border: 1.5px solid #2e2e3e !important;
    border-radius: 12px !important;
    color: #f1f1f5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}

.stTextArea textarea:focus {
    border-color: #7c5cfc !important;
    box-shadow: 0 0 0 3px rgba(124,92,252,0.2) !important;
}

.stButton button {
    background: linear-gradient(135deg, #7c5cfc, #5b3fe8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(124,92,252,0.45) !important;
}

.footer-text {
    text-align: center;
    color: #55556a;
    font-size: 12px;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #2e2e3e;
    margin-top: 2rem;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

div[data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="badge">● AI Powered</div>
    <div class="main-title">Sentiment Analyzer</div>
    <div class="subtitle">Instantly detect the emotional tone of any text<br>using a state-of-the-art transformer model.</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

clf = load_model()

txt = st.text_area("", placeholder="Type or paste any sentence here…", height=150, label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze:
    if not txt.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Running analysis…"):
            result = clf(txt[:512])[0]
            label = result["label"]
            score = result["score"]
            pct = score * 100

        if label == "POSITIVE":
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-title-pos">✅ &nbsp;Positive</div>
                <div class="result-sub">The text carries a positive sentiment.</div>
                <div class="conf-label">Confidence — {pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-title-neg">❌ &nbsp;Negative</div>
                <div class="result-sub">The text carries a negative sentiment.</div>
                <div class="conf-label">Confidence — {pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)

# Footer
st.markdown("""
<div class="footer-text">
    Built with Hugging Face Transformers &nbsp;·&nbsp; CS4106 Foundations of Generative AI
</div>
""", unsafe_allow_html=True)
