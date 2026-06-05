"""
╔══════════════════════════════════════════════════════════════╗
║          AI PLANT DOCTOR — Premium Dashboard                 ║
║  ─────────────────────────────────────────────────────────   ║
║  PREDICTION BACKEND : 100% ORIGINAL — NOT MODIFIED           ║
║  UI LAYER           : Fully redesigned premium dashboard     ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="AI Plant Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  STANDARD IMPORTS
# ─────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image
import io, base64, json, datetime, time
import google.generativeai as genai

# ═════════════════════════════════════════════════════════════
#  ██████████████████████████████████████████████████████████
#  ██                                                        ██
#  ██   ORIGINAL PREDICTION BACKEND — NOT MODIFIED          ██
#  ██   Preserved verbatim from the original working code   ██
#  ██                                                        ██
#  ██████████████████████████████████████████████████████████
# ═════════════════════════════════════════════════════════════

import tensorflow as tf

@st.cache_resource
def load_model_resource():
    """Original model loading — preserved exactly."""
    return tf.keras.models.load_model("trained_model.h5")

# Original class names — preserved exactly
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

def model_prediction(test_image):
    """
    ╔══════════════════════════════════════════════════╗
    ║  ORIGINAL model_prediction() — NOT MODIFIED     ║
    ║  target_size, normalization, preprocessing,      ║
    ║  and prediction pipeline preserved exactly.      ║
    ╚══════════════════════════════════════════════════╝
    """
    model = load_model_resource()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])   # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)   # return index + confidence

# ═════════════════════════════════════════════════════════════
#  END OF ORIGINAL PREDICTION BACKEND
# ═════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────
#  GEMINI AI CONFIGURATION
# ─────────────────────────────────────────────────────────────
import google.generativeai as genai

@st.cache_resource(show_spinner=False)
def configure_gemini():

    try:
        api_key = st.secrets["GEMINI_API_KEY"]

        genai.configure(api_key=api_key)

        return genai.GenerativeModel("gemini-2.5-flash")

    except Exception as e:
        print("Gemini Setup Error:", e)
        return None

        st.success("✅ API key found")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        st.success("✅ Gemini model initialized")

        return model

    except Exception as e:
        st.error(f"❌ Gemini Setup Error: {e}")
        return None

GEMINI_PROMPT = """
You are an expert plant pathologist AI assistant.
Detected disease: "{disease}" on plant: "{plant}"

Respond ONLY with a valid JSON object — no markdown fences, no preamble.
Use exactly these keys:
{{
  "summary": "2-3 sentence overview of this specific disease",
  "causes": ["cause 1", "cause 2", "cause 3"],
  "symptoms": ["symptom 1", "symptom 2", "symptom 3", "symptom 4"],
  "treatment": ["step 1", "step 2", "step 3"],
  "pesticides": ["product & dosage 1", "product & dosage 2", "product & dosage 3"],
  "organic": ["remedy 1", "remedy 2", "remedy 3"],
  "prevention": ["tip 1", "tip 2", "tip 3", "tip 4"],
  "severity": "Low",
  "severity_note": "one sentence explaining the severity rating"
}}
severity must be exactly one of: Low | Medium | High
"""

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_gemini_analysis(disease: str, plant: str) -> dict:
    """Cached Gemini call — same disease+plant pair reuses result for 1 hour."""

    client = configure_gemini()

    if client:
        try:
            prompt = GEMINI_PROMPT.format(
                disease=disease,
                plant=plant
            )

            response = client.generate_content(prompt)

            raw = response.text.strip()

            # Remove markdown if Gemini returns it
            if raw.startswith("```"):
                raw = raw.replace("```json", "")
                raw = raw.replace("```", "")
                raw = raw.strip()

            # Convert JSON text into Python dict
            analysis = json.loads(raw)

            # Force valid severity values
            valid_levels = ["Low", "Medium", "High"]

            if analysis.get("severity") not in valid_levels:
                text = (
                    disease.lower()
                    + " "
                    + analysis.get("summary", "").lower()
                )

                high_keywords = [
                    "virus", "blight", "severe",
                    "deadly", "critical", "rapid spread"
                ]

                medium_keywords = [
                    "spot", "mildew", "rust", "rot"
                ]

                if any(k in text for k in high_keywords):
                    analysis["severity"] = "High"

                elif any(k in text for k in medium_keywords):
                    analysis["severity"] = "Medium"

                else:
                    analysis["severity"] = "Low"

            return analysis

        except Exception as e:
            st.error(f"Gemini Error: {e}")

    return _fallback_analysis(disease, plant)

def _fallback_analysis(disease: str, plant: str) -> dict:
    """Shown only if Gemini key is absent or the API call fails."""
    return {
        "summary": (f"{disease} is a disease affecting {plant} that can reduce yield "
                    "significantly if left untreated. Early detection and prompt management are critical."),
        "causes": [
            "Fungal or bacterial pathogen infection",
            "High humidity and poor air circulation",
            "Infected soil or plant debris",
        ],
        "symptoms": [
            "Yellowing or browning of leaf tissue",
            "Water-soaked or necrotic lesions on leaf surface",
            "Premature defoliation and dieback",
            "Reduced fruit set and quality",
        ],
        "treatment": [
            "Remove and destroy all infected plant material immediately",
            "Apply appropriate fungicide or bactericide at recommended dosage",
            "Improve drainage and reduce overhead irrigation",
        ],
        "pesticides": [
            "Mancozeb 75WP — 2.5 g/litre water",
            "Copper Oxychloride 50WP — 3.0 g/litre water",
            "Chlorothalonil 75WP — 2.0 g/litre water",
        ],
        "organic": [
            "Neem oil spray (5 ml/litre) every 7 days",
            "Baking soda foliar spray (5 g/litre) as preventive measure",
            "Trichoderma-based biocontrol agents applied to soil",
        ],
        "prevention": [
            "Use certified disease-free planting material",
            "Maintain optimal plant spacing for adequate airflow",
            "Avoid excess nitrogen fertilisation",
            "Rotate crops every season to break disease cycles",
        ],
        "severity": "Medium",
        "severity_note": "Moderate yield loss (20–40%) expected without intervention within 7–10 days.",
    }


# ─────────────────────────────────────────────────────────────
#  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────
def parse_class_label(label: str):
    """'Plant___Disease' → (plant_str, disease_str)."""
    if "___" in label:
        parts = label.split("___", 1)
        plant   = parts[0].replace("_", " ").replace(",", "").strip()
        disease = parts[1].replace("_", " ").strip()
        return plant, disease
    return "Plant", label.replace("_", " ")

def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def save_uploaded_to_temp(uploaded_file) -> str:
    """Save UploadedFile bytes to a temp file; return path for keras.load_img."""
    import tempfile, os
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def build_text_report(plant, disease, conf, analysis, timestamp) -> str:
    sep = "─" * 58
    def blist(items): return "\n".join(f"  • {i}" for i in items)
    return "\n".join([
        "=" * 58,
        "      AI PLANT DOCTOR — DIAGNOSTIC REPORT",
        "=" * 58,
        f"  Generated : {timestamp}",
        f"  Plant     : {plant}",
        f"  Disease   : {disease}",
        f"  Confidence: {conf*100:.2f}%",
        f"  Severity  : {analysis.get('severity','—')}",
        sep, "  DISEASE SUMMARY", sep,
        f"  {analysis.get('summary','')}",
        "", sep, "  CAUSES", sep,   blist(analysis.get("causes",[])),
        "", sep, "  SYMPTOMS", sep, blist(analysis.get("symptoms",[])),
        "", sep, "  TREATMENT", sep, blist(analysis.get("treatment",[])),
        "", sep, "  CHEMICAL PESTICIDES", sep, blist(analysis.get("pesticides",[])),
        "", sep, "  ORGANIC / NATURAL SOLUTIONS", sep, blist(analysis.get("organic",[])),
        "", sep, "  PREVENTION TIPS", sep, blist(analysis.get("prevention",[])),
        "", sep,
        f"  SEVERITY : {analysis.get('severity','—')}",
        f"  {analysis.get('severity_note','')}",
        "", "=" * 58,
        "  AI Plant Doctor | CNN + Gemini AI",
        "=" * 58,
    ])


# ─────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:           #020c06;
  --bg2:          #061209;
  --surface:      rgba(255,255,255,0.032);
  --surface-hov:  rgba(255,255,255,0.056);
  --border:       rgba(255,255,255,0.072);
  --border-g:     rgba(34,197,94,0.26);
  --green:        #22c55e;
  --green-dim:    #15803d;
  --green-glow:   rgba(34,197,94,0.18);
  --text-hi:      #f0fdf4;
  --text-md:      #86efac;
  --text-lo:      #166534;
  --red:          #f87171;
  --amber:        #fbbf24;
  --sky:          #38bdf8;
  --purple:       #c084fc;
  --teal:         #2dd4bf;
  --orange:       #fb923c;
  --pink:         #f472b6;
  --r-lg: 20px; --r-md: 14px; --r-sm: 9px;
  --shadow: 0 8px 32px rgba(0,0,0,0.42);
  --tr: 0.24s cubic-bezier(0.4,0,0.2,1);
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text-hi) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 90% 55% at 50% -10%,rgba(34,197,94,0.13) 0%,transparent 65%),
      radial-gradient(ellipse 55% 35% at 90% 90%,rgba(21,128,61,0.09) 0%,transparent 55%),
      var(--bg) !important;
    min-height: 100vh;
}

[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="stSidebar"],#MainMenu,footer,header { display:none !important; }

section.main > div { padding-top: 0 !important; }
.block-container { max-width:1300px !important; padding:0 2rem 5rem !important; margin:0 auto !important; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:#166534; border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--green-dim); }

/* ── Hero ───────────────────────────────── */
.hero-wrap {
    position:relative; padding:4.5rem 1rem 3rem;
    text-align:center; overflow:hidden;
}
.hero-wrap::before {
    content:''; position:absolute; inset:0;
    background:
      repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(34,197,94,0.03) 39px,rgba(34,197,94,0.03) 40px),
      repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(34,197,94,0.03) 39px,rgba(34,197,94,0.03) 40px);
    pointer-events:none;
}
.hero-eyebrow {
    display:inline-flex; align-items:center; gap:8px;
    background:rgba(34,197,94,0.09); border:1px solid rgba(34,197,94,0.25);
    border-radius:100px; padding:6px 18px 6px 10px;
    font-size:0.73rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase;
    color:var(--green); margin-bottom:1.8rem;
    animation:fadeUp 0.6s ease both;
}
.eyebrow-dot {
    width:8px; height:8px; background:var(--green);
    border-radius:50%; box-shadow:0 0 8px var(--green);
    animation:blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.6)} }

.hero-title {
    font-size:clamp(3.2rem,7vw,6rem); font-weight:900;
    line-height:1.0; letter-spacing:-0.04em; margin-bottom:1rem;
    background:linear-gradient(140deg,#fff 0%,#bbf7d0 35%,#22c55e 65%,#15803d 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:fadeUp 0.65s ease 0.08s both;
}
.hero-title span {
    display:block; font-size:0.42em; font-weight:400; letter-spacing:-0.01em;
    background:linear-gradient(90deg,#86efac,#4ade80);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-top:0.35em;
}
.hero-sub {
    font-size:clamp(0.95rem,1.8vw,1.15rem); font-weight:300;
    color:#4ade80; max-width:580px; margin:0 auto 3rem;
    line-height:1.7; opacity:0.75; animation:fadeUp 0.65s ease 0.16s both;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }

.hero-stats { display:flex; justify-content:center; flex-wrap:wrap; gap:1rem; animation:fadeUp 0.65s ease 0.28s both; }
.stat-pill {
    display:flex; align-items:center; gap:10px;
    background:rgba(255,255,255,0.035); border:1px solid var(--border-g);
    border-radius:var(--r-md); padding:0.7rem 1.4rem;
    backdrop-filter:blur(8px); transition:transform var(--tr),box-shadow var(--tr);
}
.stat-pill:hover { transform:translateY(-3px); box-shadow:0 8px 28px rgba(34,197,94,0.15); }
.stat-pill-icon { font-size:1.5rem; }
.stat-pill-body { text-align:left; }
.stat-pill-num { font-size:1.3rem; font-weight:800; color:var(--green); display:block; line-height:1; }
.stat-pill-label { font-size:0.7rem; font-weight:500; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-lo); opacity:0.85; }

/* ── Dividers / section heads ───────────── */
.hr { height:1px; background:linear-gradient(90deg,transparent,rgba(34,197,94,0.20),transparent); margin:0.5rem 0 2.5rem; }
.sec-head { display:flex; align-items:center; gap:12px; margin-bottom:1.5rem; }
.sec-head-label { font-size:0.7rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase; color:var(--green); white-space:nowrap; }
.sec-head-line  { flex:1; height:1px; background:rgba(34,197,94,0.13); }

/* ── File uploader ──────────────────────── */
[data-testid="stFileUploader"] > div:first-child {
    background:rgba(34,197,94,0.025) !important;
    border:1.5px dashed rgba(34,197,94,0.30) !important;
    border-radius:var(--r-lg) !important; padding:3rem 2rem !important;
    transition:border-color var(--tr),background var(--tr) !important;
}
[data-testid="stFileUploader"] > div:first-child:hover {
    border-color:rgba(34,197,94,0.55) !important;
    background:rgba(34,197,94,0.045) !important;
}

/* FIX: Targeted only the actual instruction text wrapper to avoid breaking the delete button */
[data-testid="stFileUploaderDropzoneInstructions"] > div,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small { 
    color:#4ade80 !important; 
    font-family:'Outfit',sans-serif !important; 
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { display:none; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content:'🌿'; font-size:2.5rem; display:block; margin-bottom:0.6rem; }

/* FIX: Upload button text and icon color restore to Black */
[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] {
    color: #000000 !important;
}
[data-testid="stFileUploader"] button[data-testid="stBaseButton-secondary"] svg {
    display: inline-block !important; /* Restore button icon */
    fill: #000000 !important;
    color: #000000 !important;
}

/* FIX: Correct color and look for the file remove/clear button (No more black dot) */
[data-testid="stFileUploaderDeleteBtn"] {
    background-color: rgba(248, 113, 113, 0.1) !important;
    border-radius: 50% !important;
}
[data-testid="stFileUploaderDeleteBtn"] svg {
    display: inline-block !important;
    fill: #f87171 !important; /* Soft red color for clear icon */
    color: #f87171 !important;
}
[data-testid="stFileUploaderDeleteBtn"]:hover {
    background-color: rgba(248, 113, 113, 0.2) !important;
}

/* ── Preview grid ───────────────────────── */
.preview-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:1rem; margin:1.5rem 0; }
.preview-card {
    background:var(--surface); border:1px solid var(--border); border-radius:var(--r-md);
    overflow:hidden; transition:transform var(--tr),box-shadow var(--tr),border-color var(--tr); cursor:default;
}
.preview-card:hover { transform:translateY(-5px) scale(1.02); box-shadow:0 16px 40px rgba(0,0,0,0.50); border-color:var(--border-g); }
.preview-card img   { width:100%; aspect-ratio:1; object-fit:cover; display:block; }
.preview-card-name  { padding:7px 10px; font-size:0.68rem; font-family:'JetBrains Mono',monospace; color:#166534; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; border-top:1px solid var(--border); }

/* ── Buttons ────────────────────────────── */
div.stButton > button {
    background:linear-gradient(135deg,#16a34a 0%,#15803d 50%,#166534 100%) !important;
    color:#f0fdf4 !important; border:none !important; border-radius:var(--r-md) !important;
    padding:0.8rem 2.4rem !important; font-family:'Outfit',sans-serif !important;
    font-weight:700 !important; font-size:0.97rem !important; letter-spacing:0.02em !important;
    transition:all var(--tr) !important;
    box-shadow:0 4px 20px rgba(34,197,94,0.22),inset 0 1px 0 rgba(255,255,255,0.08) !important;
    width:100% !important;
}
div.stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 10px 36px rgba(34,197,94,0.38) !important; filter:brightness(1.08) !important; }
div.stButton > button:active { transform:translateY(0) !important; }

div.stDownloadButton > button {
    background:rgba(34,197,94,0.08) !important; color:var(--green) !important;
    border:1px solid rgba(34,197,94,0.28) !important; border-radius:var(--r-sm) !important;
    font-family:'Outfit',sans-serif !important; font-weight:600 !important; font-size:0.84rem !important;
    transition:all var(--tr) !important;
}
div.stDownloadButton > button:hover { background:rgba(34,197,94,0.16) !important; border-color:rgba(34,197,94,0.50) !important; transform:translateY(-1px) !important; }

/* ── Result card ─────────────────────────── */
.res-card {
    background:var(--surface); border:1px solid var(--border); border-radius:var(--r-lg);
    padding:1.8rem; margin-bottom:2rem; position:relative; overflow:hidden;
    animation:revealCard 0.5s ease both; box-shadow:var(--shadow);
}
.res-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--green),transparent); }
@keyframes revealCard { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }

.res-plant-tag { font-size:0.7rem; font-weight:700; letter-spacing:0.13em; text-transform:uppercase; color:var(--green); margin-bottom:0.3rem; }
.res-disease   { font-size:clamp(1.4rem,3vw,1.9rem); font-weight:800; line-height:1.1; color:var(--text-hi); letter-spacing:-0.02em; margin-bottom:0.6rem; }
.res-ts        { font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:var(--text-lo); margin-bottom:1.2rem; }

.badge { display:inline-flex; align-items:center; gap:6px; padding:5px 14px; border-radius:100px; font-size:0.73rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:1.4rem; }
.badge-ok   { background:rgba(34,197,94,0.12);   border:1px solid rgba(34,197,94,0.32);   color:#4ade80; }
.badge-warn { background:rgba(248,113,113,0.10);  border:1px solid rgba(248,113,113,0.28); color:#f87171; }
.badge-dot  { width:7px; height:7px; border-radius:50%; }
.badge-ok   .badge-dot { background:#4ade80; }
.badge-warn .badge-dot { background:#f87171; }

.conf-row   { display:flex; justify-content:space-between; align-items:center; margin-bottom:5px; }
.conf-title { font-size:0.75rem; color:var(--text-lo); font-weight:500; text-transform:uppercase; letter-spacing:0.08em; }
.conf-pct   { font-size:0.88rem; font-weight:800; color:var(--green); }
.conf-bg    { height:7px; background:rgba(255,255,255,0.06); border-radius:100px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:100px; background:linear-gradient(90deg,#15803d,#22c55e,#86efac); box-shadow:0 0 10px rgba(34,197,94,0.40); }

/* ── Mini metrics row ───────────────────── */
.mini-metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:0.8rem; margin:1.8rem 0; }
.mini-metric  {
    background:var(--surface); border:1px solid var(--border); border-radius:var(--r-md);
    padding:1rem; text-align:center; transition:border-color var(--tr),background var(--tr);
}
.mini-metric:hover { border-color:var(--border-g); background:var(--surface-hov); }
.mm-icon  { font-size:1.5rem; display:block; margin-bottom:0.4rem; }
.mm-val   { font-size:1.25rem; font-weight:800; color:var(--text-md); display:block; }
.mm-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-lo); font-weight:600; }

/* ── Gemini cards grid ──────────────────── */
.gem-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1.5rem; }
.gem-card {
    background:var(--surface); border:1px solid var(--border); border-radius:var(--r-md);
    padding:1.4rem; transition:transform var(--tr),box-shadow var(--tr),border-color var(--tr);
    animation:revealCard 0.45s ease both; position:relative; overflow:hidden;
}
.gem-card::after { content:''; position:absolute; inset:0; background:linear-gradient(135deg,rgba(255,255,255,0.015) 0%,transparent 60%); pointer-events:none; }
.gem-card:hover { transform:translateY(-4px); box-shadow:0 14px 44px rgba(0,0,0,0.45); }
.gem-card.full { grid-column:1/-1; }

.gem-card.c-summary    { border-top:2px solid #22c55e; }
.gem-card.c-causes     { border-top:2px solid #fbbf24; }
.gem-card.c-symptoms   { border-top:2px solid #f87171; }
.gem-card.c-treatment  { border-top:2px solid #38bdf8; }
.gem-card.c-pesticides { border-top:2px solid #c084fc; }
.gem-card.c-organic    { border-top:2px solid #2dd4bf; }
.gem-card.c-prevention { border-top:2px solid #fb923c; }
.gem-card.c-severity   { border-top:2px solid #f472b6; }

.gem-header { display:flex; align-items:center; gap:9px; margin-bottom:1rem; }
.gem-icon   { font-size:1.3rem; }
.gem-title  { font-size:0.88rem; font-weight:700; color:var(--text-hi); letter-spacing:0.01em; }
.gem-body   { font-size:0.85rem; color:#4ade80; line-height:1.8; opacity:0.82; }
.gem-body ul { padding-left:1.1rem; margin:0; }
.gem-body li { margin-bottom:0.3rem; }
.gem-body strong { color:var(--text-md); font-weight:600; opacity:1; }

.sev-chip { display:inline-block; padding:5px 20px; border-radius:100px; font-weight:900; font-size:1rem; letter-spacing:0.05em; margin-bottom:0.6rem; }
.sev-Low    { background:rgba(34,197,94,0.14);   color:#4ade80;  border:1px solid rgba(34,197,94,0.30); }
.sev-Medium { background:rgba(251,191,36,0.13);  color:#fbbf24;  border:1px solid rgba(251,191,36,0.28); }
.sev-High   { background:rgba(248,113,113,0.13); color:#f87171;  border:1px solid rgba(248,113,113,0.28); }

/* ── Tabs ───────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.025) !important; border-radius:var(--r-md) !important;
    padding:4px !important; gap:3px !important; border:1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background:transparent !important; border-radius:10px !important; color:#166534 !important;
    font-family:'Outfit',sans-serif !important; font-weight:600 !important;
    font-size:0.87rem !important; padding:0.5rem 1.4rem !important; transition:all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] { background:rgba(34,197,94,0.15) !important; color:#4ade80 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }

/* ── History ─────────────────────────────── */
.hist-item {
    display:flex; align-items:center; justify-content:space-between;
    padding:0.9rem 1.2rem; background:var(--surface); border:1px solid var(--border);
    border-radius:var(--r-md); margin-bottom:0.6rem;
    transition:border-color var(--tr),background var(--tr);
}
.hist-item:hover { border-color:var(--border-g); background:var(--surface-hov); }
.hist-name  { font-size:0.88rem; font-weight:600; color:var(--text-md); }
.hist-meta  { font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:var(--text-lo); margin-top:2px; }
.hist-right { text-align:right; }
.hist-conf  { font-size:0.92rem; font-weight:800; color:var(--green); display:block; }
.hist-badge { display:inline-block; font-size:0.65rem; font-weight:700; padding:2px 9px; border-radius:100px; letter-spacing:0.07em; }
.hist-ok    { background:rgba(34,197,94,0.12);   color:#4ade80; }
.hist-warn  { background:rgba(248,113,113,0.10);  color:#f87171; }

/* ── Progress bar ───────────────────────── */
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#166534,#22c55e) !important; border-radius:100px !important; }
[data-testid="stProgressBar"] > div { background:rgba(255,255,255,0.05) !important; border-radius:100px !important; }

/* ── Expander ───────────────────────────── */
[data-testid="stExpander"] { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:var(--r-md) !important; }
[data-testid="stExpander"] summary { font-family:'Outfit',sans-serif !important; font-weight:600 !important; color:#4ade80 !important; }

/* ── Alert / spinner ────────────────────── */
[data-testid="stAlert"] { background:rgba(34,197,94,0.07) !important; border:1px solid rgba(34,197,94,0.20) !important; border-radius:var(--r-md) !important; color:#86efac !important; font-family:'Outfit',sans-serif !important; }
[data-testid="stSpinner"] p { color:#4ade80 !important; font-family:'Outfit',sans-serif !important; font-size:0.9rem !important; }
[data-testid="stImage"] img { border-radius:var(--r-md) !important; border:1px solid var(--border) !important; }

/* ── Analytics bars ─────────────────────── */
.a-bar-bg   { background:rgba(255,255,255,0.05); border-radius:100px; height:8px; overflow:hidden; margin-top:5px; margin-bottom:1rem; }
.a-bar-fill { height:100%; border-radius:100px; box-shadow:0 0 10px rgba(34,197,94,0.30); }

/* ── Footer ─────────────────────────────── */
.footer      { margin-top:5rem; padding:2.5rem 0 1rem; border-top:1px solid rgba(34,197,94,0.09); text-align:center; }
.footer-brand { font-size:1.15rem; font-weight:900; background:linear-gradient(90deg,#22c55e,#86efac); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:0.4rem; }
.footer-copy  { font-size:0.75rem; color:#052e16; }

/* ── Empty state ─────────────────────────── */
.empty-state { text-align:center; padding:5rem 1rem; color:var(--text-lo); }
.empty-state .icon  { font-size:4rem; display:block; margin-bottom:1rem; }
.empty-state .title { font-size:1.05rem; font-weight:700; color:#166534; margin-bottom:0.4rem; }
.empty-state .hint  { font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  UI COMPONENT RENDERERS
# ─────────────────────────────────────────────────────────────

def render_hero():
    n = len(st.session_state.get("history", []))
    st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-eyebrow">
    <span class="eyebrow-dot"></span>
    Deep Learning · Gemini AI · Precision Agriculture
  </div>
  <div class="hero-title">
    AI Plant Doctor
    <span>AI Powered Plant Disease Detection &amp; Treatment System</span>
  </div>
  <div class="hero-sub">
    Upload leaf images — our CNN instantly identifies diseases<br>
    and Gemini AI generates complete treatment guidance.
  </div>
  <div class="hero-stats">
    <div class="stat-pill">
      <span class="stat-pill-icon">🧬</span>
      <div class="stat-pill-body">
        <span class="stat-pill-num">38</span>
        <span class="stat-pill-label">Disease Classes</span>
      </div>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-icon">🤖</span>
      <div class="stat-pill-body">
        <span class="stat-pill-num">AI</span>
        <span class="stat-pill-label">Treatment Guide</span>
      </div>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-icon">📸</span>
      <div class="stat-pill-body">
        <span class="stat-pill-num">Multi</span>
        <span class="stat-pill-label">Image Upload</span>
      </div>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-icon">🔍</span>
      <div class="stat-pill-body">
        <span class="stat-pill-num">{n}</span>
        <span class="stat-pill-label">Total Scans</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

def section_head(label):
    st.markdown(f"""
<div class="sec-head">
  <span class="sec-head-label">{label}</span>
  <span class="sec-head-line"></span>
</div>""", unsafe_allow_html=True)

def render_preview_grid(files):
    html = ""
    for f in files:
        img  = Image.open(f).convert("RGB")
        b64  = pil_to_b64(img)
        name = (f.name[:20] + "…") if len(f.name) > 20 else f.name
        html += f'<div class="preview-card"><img src="data:image/jpeg;base64,{b64}" /><div class="preview-card-name">📄 {name}</div></div>'
    st.markdown(f'<div class="preview-grid">{html}</div>', unsafe_allow_html=True)

def render_result_card(plant, disease, conf, timestamp, is_healthy):
    bc  = "badge-ok" if is_healthy else "badge-warn"
    bt  = "✓ Healthy" if is_healthy else "⚠ Disease Detected"
    pct = conf * 100
    st.markdown(f"""
<div class="res-card">
  <div class="res-plant-tag">🌱 {plant}</div>
  <div class="res-disease">{disease}</div>
  <div class="res-ts">⏱ {timestamp}</div>
  <span class="badge {bc}"><span class="badge-dot"></span>{bt}</span>
  <div class="conf-row">
    <span class="conf-title">Confidence Score</span>
    <span class="conf-pct">{pct:.2f}%</span>
  </div>
  <div class="conf-bg"><div class="conf-fill" style="width:{pct:.2f}%"></div></div>
</div>""", unsafe_allow_html=True)

def render_gemini_cards(analysis):
    def ul(items):
        if not items: return "<em style='opacity:0.5'>No data</em>"
        return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    sev = analysis.get("severity", "Medium")
    sev_e = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(sev,"🟡")
    st.markdown(f"""
<div class="gem-grid">
  <div class="gem-card c-summary full">
    <div class="gem-header"><span class="gem-icon">🔬</span><span class="gem-title">Disease Overview</span></div>
    <div class="gem-body">{analysis.get("summary","—")}</div>
  </div>
  <div class="gem-card c-causes">
    <div class="gem-header"><span class="gem-icon">⚡</span><span class="gem-title">Causes</span></div>
    <div class="gem-body">{ul(analysis.get("causes",[]))}</div>
  </div>
  <div class="gem-card c-symptoms">
    <div class="gem-header"><span class="gem-icon">🩺</span><span class="gem-title">Symptoms</span></div>
    <div class="gem-body">{ul(analysis.get("symptoms",[]))}</div>
  </div>
  <div class="gem-card c-treatment">
    <div class="gem-header"><span class="gem-icon">💊</span><span class="gem-title">Treatment Steps</span></div>
    <div class="gem-body">{ul(analysis.get("treatment",[]))}</div>
  </div>
  <div class="gem-card c-pesticides">
    <div class="gem-header"><span class="gem-icon">🧪</span><span class="gem-title">Chemical Pesticides</span></div>
    <div class="gem-body">{ul(analysis.get("pesticides",[]))}</div>
  </div>
  <div class="gem-card c-organic">
    <div class="gem-header"><span class="gem-icon">🌿</span><span class="gem-title">Organic / Natural Cure</span></div>
    <div class="gem-body">{ul(analysis.get("organic",[]))}</div>
  </div>
  <div class="gem-card c-prevention">
    <div class="gem-header"><span class="gem-icon">🛡️</span><span class="gem-title">Prevention Tips</span></div>
    <div class="gem-body">{ul(analysis.get("prevention",[]))}</div>
  </div>
  <div class="gem-card c-severity">
    <div class="gem-header"><span class="gem-icon">{sev_e}</span><span class="gem-title">Severity Level</span></div>
    <div class="gem-body">
      <span class="sev-chip sev-{sev}">{sev}</span><br><br>
      {analysis.get("severity_note","—")}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

def render_history():
    h = st.session_state.get("history", [])
    if not h:
        st.markdown('<div class="empty-state" style="padding:2rem 0"><span class="icon">📋</span><div class="title">No scans yet</div><div class="hint">History appears here after scanning images.</div></div>', unsafe_allow_html=True)
        return
    for item in reversed(h[-15:]):
        bc = "hist-ok" if item["is_healthy"] else "hist-warn"
        bt = "Healthy" if item["is_healthy"] else "Diseased"
        st.markdown(f"""
<div class="hist-item">
  <div>
    <div class="hist-name">🌱 {item["plant"]} — {item["disease"]}</div>
    <div class="hist-meta">{item["timestamp"]}</div>
  </div>
  <div class="hist-right">
    <span class="hist-conf">{item["conf"]*100:.1f}%</span>
    <span class="hist-badge {bc}">{bt}</span>
  </div>
</div>""", unsafe_allow_html=True)

def render_analytics():
    h = st.session_state.get("history", [])
    if not h: return
    total   = len(h)
    n_dis   = sum(1 for x in h if not x["is_healthy"])
    n_hlth  = total - n_dis
    avg_c   = sum(x["conf"] for x in h) / total
    dp      = (n_dis  / total) * 100
    hp      = (n_hlth / total) * 100
    st.markdown(f"""
<div class="res-card" style="margin-top:1.2rem">
  <div style="font-weight:800;font-size:1rem;color:var(--text-md);margin-bottom:1.2rem">📈 Session Analytics</div>
  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-lo);font-weight:600;margin-bottom:4px">Disease Rate <span style="color:#f87171">{dp:.0f}%</span></div>
  <div class="a-bar-bg"><div class="a-bar-fill" style="width:{dp:.0f}%;background:linear-gradient(90deg,#f87171,#fbbf24)"></div></div>
  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-lo);font-weight:600;margin-bottom:4px">Healthy Rate <span style="color:#4ade80">{hp:.0f}%</span></div>
  <div class="a-bar-bg"><div class="a-bar-fill" style="width:{hp:.0f}%;background:linear-gradient(90deg,#15803d,#22c55e)"></div></div>
  <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text-lo);font-weight:600;margin-bottom:4px">Avg Confidence <span style="color:#38bdf8">{avg_c*100:.1f}%</span></div>
  <div class="a-bar-bg"><div class="a-bar-fill" style="width:{avg_c*100:.1f}%;background:linear-gradient(90deg,#0369a1,#38bdf8)"></div></div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    inject_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "results" not in st.session_state:
        st.session_state.results = []

    render_hero()
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Session metric pills ───────────────────────────────
    h = st.session_state.history
    nd = sum(1 for x in h if not x["is_healthy"])
    st.markdown(f"""
<div class="mini-metrics">
  <div class="mini-metric"><span class="mm-icon">🔍</span><span class="mm-val">{len(h)}</span><span class="mm-label">Total Scans</span></div>
  <div class="mini-metric"><span class="mm-icon">⚠️</span><span class="mm-val">{nd}</span><span class="mm-label">Diseased</span></div>
  <div class="mini-metric"><span class="mm-icon">✅</span><span class="mm-val">{len(h)-nd}</span><span class="mm-label">Healthy</span></div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    tab_scan, tab_hist, tab_about = st.tabs(["🔬  Scan & Diagnose", "📋  Scan History", "ℹ️  About"])

    # ════════════════════════════════════════
    #  SCAN TAB
    # ════════════════════════════════════════
    with tab_scan:
        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        section_head("📤  Upload Plant Leaf Images")

        uploaded_files = st.file_uploader(
            "Drag & drop images here, or click to browse",
            type=["jpg","jpeg","png","webp","bmp"],
            accept_multiple_files=True,
            label_visibility="visible",
        )

        if uploaded_files:
            cnt = len(uploaded_files)
            st.markdown(f'<div style="color:#166534;font-size:0.8rem;margin:0.4rem 0 0.2rem">{cnt} image{"s" if cnt>1 else ""} ready for analysis</div>', unsafe_allow_html=True)
            render_preview_grid(uploaded_files)

            col_btn, col_clr = st.columns([4, 1], gap="small")
            with col_btn:
                analyse = st.button(f"🔬  Analyse {cnt} Image{'s' if cnt>1 else ''}", use_container_width=True)
            with col_clr:
                cleared = st.button("🗑️ Clear", use_container_width=True)

            if cleared:
                st.session_state.results = []
                st.rerun()

            if analyse:
                results   = []
                prog      = st.progress(0, text="Initialising…")

                for i, f in enumerate(uploaded_files):
                    prog.progress(int(i / cnt * 100), text=f"Analysing {f.name} ({i+1}/{cnt})…")

                    # ─────────────────────────────────────
                    #  ORIGINAL PREDICTION — NOT MODIFIED
                    # ─────────────────────────────────────
                    tmp_path           = save_uploaded_to_temp(f)
                    pred_idx, confidence = model_prediction(tmp_path)
                    predicted_label    = class_name[pred_idx]
                    # ─────────────────────────────────────

                    plant, disease = parse_class_label(predicted_label)
                    is_healthy     = "healthy" in disease.lower()
                    ts             = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

                    with st.spinner(f"🤖 Gemini generating insights for {f.name}…"):
                        analysis = fetch_gemini_analysis(disease, plant)

                    f.seek(0)
                    pil_img = Image.open(f).convert("RGB")

                    results.append({
                        "filename"  : f.name,
                        "pil_img"   : pil_img,
                        "plant"     : plant,
                        "disease"   : disease,
                        "confidence": float(confidence),
                        "is_healthy": is_healthy,
                        "timestamp" : ts,
                        "analysis"  : analysis,
                    })

                    st.session_state.history.append({
                        "plant"     : plant,
                        "disease"   : disease,
                        "conf"      : float(confidence),
                        "is_healthy": is_healthy,
                        "timestamp" : ts,
                    })

                prog.progress(100, text="✅ Analysis complete!")
                time.sleep(0.5)
                prog.empty()
                st.session_state.results = results
                st.rerun()

        # ── Render saved results ───────────────────────────
        results = st.session_state.get("results", [])
        if results:
            st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
            section_head(f"📊  Diagnosis Results — {len(results)} Image{'s' if len(results)>1 else ''}")

            for idx, res in enumerate(results):
                col_img, col_info = st.columns([1, 2], gap="large")
                with col_img:
                    st.image(res["pil_img"], caption=res["filename"], width=280)
                with col_info:
                    render_result_card(
                        res["plant"], res["disease"],
                        res["confidence"], res["timestamp"], res["is_healthy"],
                    )
                    rpt = build_text_report(
                        res["plant"], res["disease"],
                        res["confidence"], res["analysis"], res["timestamp"],
                    )
                    safe = res["plant"].lower().replace(" ","_")
                    st.download_button(
                        "⬇️  Download Diagnostic Report", rpt,
                        file_name=f"plant_report_{safe}_{idx+1}.txt",
                        mime="text/plain", use_container_width=True,
                    )

                if res["analysis"]:
                    st.markdown('<div style="height:0.3rem"></div>', unsafe_allow_html=True)
                    section_head("🤖  Gemini AI Treatment Plan")
                    render_gemini_cards(res["analysis"])

                if idx < len(results) - 1:
                    st.markdown('<div class="hr" style="margin:2.5rem 0"></div>', unsafe_allow_html=True)

        elif not uploaded_files:
            st.markdown("""
<div class="empty-state">
  <span class="icon">🌿</span>
  <div class="title">Upload plant leaf images to begin</div>
  <div class="hint">Supports JPG · PNG · WEBP · BMP &nbsp;|&nbsp; Multi-image batch analysis</div>
</div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    #  HISTORY TAB
    # ════════════════════════════════════════
    with tab_hist:
        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        section_head("🕐  Recent Scan History")
        render_history()
        if st.session_state.history:
            render_analytics()
            _, col_clr = st.columns([3,1])
            with col_clr:
                if st.button("🗑️  Clear History", use_container_width=True):
                    st.session_state.history = []
                    st.session_state.results = []
                    st.rerun()

    # ════════════════════════════════════════
    #  ABOUT TAB
    # ════════════════════════════════════════
    with tab_about:
        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="res-card">
  <div style="font-size:1.5rem;font-weight:900;color:#f0fdf4;margin-bottom:0.8rem">🌿 About AI Plant Doctor</div>
  <div style="color:#4ade80;font-size:0.92rem;line-height:1.85;opacity:0.8;margin-bottom:1.5rem">
    AI Plant Doctor fuses a trained CNN with Google Gemini AI to deliver instant, accurate plant
    disease detection and expert treatment recommendations — built for farmers, agronomists,
    and researchers.
  </div>
  <div class="gem-grid">
    <div class="gem-card c-summary">
      <div class="gem-header"><span class="gem-icon">🧠</span><span class="gem-title">CNN Model</span></div>
      <div class="gem-body">PlantVillage dataset · 87,000+ leaf images · {len(class_name)} classes · Input: 128×128 px.</div>
    </div>
    <div class="gem-card c-treatment">
      <div class="gem-header"><span class="gem-icon">✨</span><span class="gem-title">Gemini AI</span></div>
      <div class="gem-body">Gemini 2.5 Flash generates structured JSON treatment plans — causes, symptoms, pesticides, organic remedies, prevention & severity.</div>
    </div>
    <div class="gem-card c-organic">
      <div class="gem-header"><span class="gem-icon">🚀</span><span class="gem-title">Tech Stack</span></div>
      <div class="gem-body">Streamlit · TensorFlow/Keras · Google Generative AI · PIL · NumPy · Python 3.10+</div>
    </div>
    <div class="gem-card c-prevention">
      <div class="gem-header"><span class="gem-icon">🌍</span><span class="gem-title">Impact</span></div>
      <div class="gem-body">Empowers smallholder farmers with AI crop intelligence — reducing preventable yield losses up to 40%.</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with st.expander("⚙️ Model & Prediction Details"):
            h_cls = sum(1 for c in class_name if 'healthy' in c)
            d_cls = len(class_name) - h_cls
            st.markdown(f"""
**Model file:** `trained_model.h5`  
**Input size:** 128 × 128 px *(target_size preserved exactly)*  
**Preprocessing:** `tf.keras.preprocessing.image.load_img` + `img_to_array` + batch expand *(preserved exactly)*  
**Confidence:** `np.max(predictions)` *(preserved exactly)*  
**Class index:** `np.argmax(predictions)` *(preserved exactly)*  
**Classes:** {len(class_name)} total — {h_cls} healthy variants, {d_cls} disease classes  
**Model caching:** `@st.cache_resource` — loaded once per session  
**Gemini caching:** `@st.cache_data(ttl=3600)` — same disease+plant pair reuses result  
""")

    # ── Footer ─────────────────────────────────────────────
    st.markdown("""
<div class="footer">
  <div class="footer-brand">🌿 AI Plant Doctor</div>
  <div class="footer-copy">Powered by CNN Deep Learning &amp; Google Gemini AI &nbsp;·&nbsp; Built with Streamlit &nbsp;·&nbsp; © 2025</div>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()