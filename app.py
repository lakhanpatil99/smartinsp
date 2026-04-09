"""
╔══════════════════════════════════════════════════════════════════════╗
║   AI-Powered Smart Quality Inspection System — v4.0                  ║
║   Enterprise Internal Tool — Fully Refactored per DIDI Report        ║
║   DIDI Fixes: P0 (stale state, audit trail), P1 (auto-analyze,      ║
║   image preview, non-deterministic AI, modular architecture)         ║
║   Run: streamlit run app.py                                          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import csv
import io
import hashlib
import random
import os
import html as html_lib  # DIDI FIX: HTML sanitization


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 1: CONFIGURATION
# ══════════════════════════════════════════════════════════════════

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "quality_data.csv")
RESULTS_DIR = os.path.join(APP_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "inspections.csv")

REQUIRED_COLUMNS = [
    "batch_id", "product_name", "inspector", "production_line",
    "shift", "date", "measurement_value", "previous_defect",
    "previous_inspections", "previous_pass_rate", "remarks",
]

st.set_page_config(
    page_title="Smart Quality Inspection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 2: DATA LAYER
# ══════════════════════════════════════════════════════════════════
# Future: Replace load_data() with Databricks Delta Lake query
#   spark.read.format("delta").load("dbfs:/quality/inspections")
# Future: Replace fetch_batch_data() with SAP ERP REST API
#   requests.get(f"{SAP_ENDPOINT}/batch/{batch_id}")
# ══════════════════════════════════════════════════════════════════

def _safe(val: str) -> str:
    """Sanitize a string for safe HTML rendering. (DIDI FIX: Issue #09)"""
    if pd.isna(val):
        return "N/A"
    return html_lib.escape(str(val).strip())


@st.cache_data(ttl=300)  # DIDI FIX: Cache TTL of 5 minutes
def load_data() -> pd.DataFrame:
    """
    Load and validate the quality inspection dataset.

    Future: Databricks SQL Connector / Delta Lake query here.
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error("❌ `quality_data.csv` not found. Place it next to `app.py`.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {_safe(str(e))}")
        return pd.DataFrame()

    # DIDI FIX: Schema validation (Issue #10)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"❌ CSV is missing columns: {', '.join(missing)}")
        return pd.DataFrame()

    df["batch_id"] = df["batch_id"].astype(str).str.strip()
    return df


def fetch_batch_data(df: pd.DataFrame, batch_id: str) -> dict | None:
    """
    Fetch a single batch record from the dataset.

    Future: Replace with Databricks SQL / SAP ERP API call.
    """
    if df.empty or not batch_id:
        return None

    try:
        match = df[df["batch_id"] == batch_id]
        if match.empty:
            return None
        r = match.iloc[0]
        return {
            "product_name": _safe(r.get("product_name", "N/A")),
            "inspector": _safe(r.get("inspector", "N/A")),
            "line": _safe(r.get("production_line", "N/A")),
            "shift": _safe(r.get("shift", "N/A")),
            "date": _safe(r.get("date", date.today().isoformat())),
            "measurement": float(r.get("measurement_value", 0.0)),
            "prev_defect": _safe(r.get("previous_defect", "None")),
            "prev_inspections": int(r.get("previous_inspections", 0)),
            "prev_pass_rate": float(r.get("previous_pass_rate", 0.0)),
            "remarks": _safe(r.get("remarks", "")),
        }
    except Exception:
        return None


def compute_kpis(df: pd.DataFrame, session_history: list) -> dict:
    """
    Compute KPIs blending CSV data with live session metrics.

    Future: Pull live KPIs from Databricks SQL analytics.
    """
    # ── CSV-based (historical) ──
    total_csv = len(df) if not df.empty else 0
    defect_rows = df[df["previous_defect"].str.strip().str.lower() != "none"] if not df.empty else pd.DataFrame()
    csv_defects = len(defect_rows)
    csv_pass = round(((total_csv - csv_defects) / total_csv) * 100, 1) if total_csv > 0 else 0.0
    critical_types = ["Crack", "Misalignment", "Dimensional Deviation"]
    csv_critical = len(defect_rows[defect_rows["previous_defect"].isin(critical_types)]) if not defect_rows.empty else 0

    # ── Session-based (live) — DIDI FIX: Issue #06 ──
    sess_total = len(session_history)
    sess_defects = sum(1 for r in session_history if r.get("defect_type", "None") != "None")
    sess_critical = sum(1 for r in session_history if r.get("risk_level") == "High")

    return {
        "total_csv": total_csv,
        "csv_pass_rate": csv_pass,
        "csv_defects": csv_defects,
        "csv_critical": csv_critical,
        "sess_total": sess_total,
        "sess_defects": sess_defects,
        "sess_critical": sess_critical,
    }


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 3: AI SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════
# Future: Replace simulate_ai_analysis() with Databricks ML Serving
#   requests.post(DATABRICKS_ML_ENDPOINT, json=payload)
# Future: Computer Vision model for image-based defect detection
#   requests.post(CV_MODEL_ENDPOINT, files={"image": img_bytes})
# ══════════════════════════════════════════════════════════════════

DEFECT_TYPES = ["None", "Crack", "Scratch", "Misalignment", "Contamination", "Dimensional Deviation"]

DEFECT_PROFILES = {
    "None": {
        "risk": "Low", "icon": "🟢", "color": "#00E676",
        "conf_range": (95.0, 99.8),
        "explanation": (
            "Comprehensive AI analysis confirms this unit meets all quality specifications. "
            "Surface integrity, dimensional accuracy, and material consistency are within "
            "acceptable tolerance bands. Cleared for the next production stage."
        ),
        "action": (
            "✅ <b>Approved for Release</b> — No further action required. Log results and "
            "proceed with standard workflow. Unit cleared for packaging &amp; dispatch."
        ),
        "action_class": "success",
    },
    "Crack": {
        "risk": "High", "icon": "🔴", "color": "#FF5252",
        "conf_range": (91.0, 98.5),
        "explanation": (
            "AI detection has identified a structural crack-type defect. Cracks are "
            "<b>critical structural integrity concerns</b> — they propagate under thermal "
            "cycling and operational stress, risking catastrophic failure. Historical data "
            "shows a <b>73% probability of worsening</b> if unaddressed."
        ),
        "action": (
            "🚨 <b>Immediate Action Required</b> — Quarantine the batch for secondary "
            "inspection. Halt the production line until RCA is completed. Initiate a "
            "Non-Conformance Report (NCR). Consider full batch recall."
        ),
        "action_class": "danger",
    },
    "Scratch": {
        "risk": "Medium", "icon": "🟡", "color": "#FFD54F",
        "conf_range": (84.0, 95.0),
        "explanation": (
            "Surface-level scratch anomaly detected. While scratches typically don't compromise "
            "structural integrity, they may affect <b>aesthetic quality, coating adhesion, and "
            "corrosion resistance</b>. Depth analysis suggests handling friction or tooling wear."
        ),
        "action": (
            "⚠️ <b>Review Required</b> — Flag for cosmetic quality review. Inspect line for "
            "tooling wear. Reclassify to High risk if scratch depth exceeds coating specs."
        ),
        "action_class": "warning",
    },
    "Misalignment": {
        "risk": "High", "icon": "🔴", "color": "#FF5252",
        "conf_range": (89.0, 97.0),
        "explanation": (
            "Dimensional analysis detected a misalignment defect — deviation from assembly specs. "
            "This can cause <b>improper fitment, mechanical stress, premature wear, and safety "
            "hazards</b>. Commonly linked to fixture calibration drift."
        ),
        "action": (
            "🚨 <b>Immediate Corrective Action</b> — Stop the assembly line, recalibrate "
            "fixtures. Check the last 50 units. Engage process engineering."
        ),
        "action_class": "danger",
    },
    "Contamination": {
        "risk": "Medium", "icon": "🟡", "color": "#FFD54F",
        "conf_range": (80.0, 93.0),
        "explanation": (
            "Foreign particle contamination detected. This can compromise <b>electrical "
            "conductivity, seal integrity, and coating adhesion</b>. Likely metallic debris "
            "or dust ingress from inadequate cleanroom protocols."
        ),
        "action": (
            "⚠️ <b>Cleaning &amp; Re-inspection Required</b> — Route to cleaning station "
            "per SOP-QC-204. Review cleanroom filtration logs."
        ),
        "action_class": "warning",
    },
    "Dimensional Deviation": {
        "risk": "High", "icon": "🔴", "color": "#FF5252",
        "conf_range": (88.0, 96.0),
        "explanation": (
            "Precision measurement identified dimensional deviation beyond tolerance. "
            "Impacts <b>fitment, performance, and interchangeability</b>. Pattern suggests "
            "CNC tool compensation issues or material lot variability."
        ),
        "action": (
            "🚨 <b>Production Hold</b> — Quarantine units. Initiate MSA to verify gauge "
            "accuracy. Review CNC parameters. FAI on next run before resuming."
        ),
        "action_class": "danger",
    },
}


def simulate_ai_analysis(batch_id: str, batch_data: dict, has_image: bool) -> dict:
    """
    Simulate AI-powered defect detection.

    DIDI FIX (Issue #02): Uses time-based variation so the same batch
    can produce different results on re-analysis, while remaining
    semi-realistic by weighting based on previous defect history.

    Future: Replace body with Databricks ML Serving API call.
    """
    # DIDI FIX: Non-deterministic — mix batch_id + current minute
    time_factor = datetime.now().strftime("%Y%m%d%H%M")
    seed = int(hashlib.md5(f"{batch_id}:{time_factor}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    prev_defect = batch_data.get("prev_defect", "None").strip()
    measurement = batch_data.get("measurement", 0.0)

    # ── Smart detection weighted by history + measurement ──
    if prev_defect != "None" and prev_defect in DEFECT_PROFILES:
        if rng.random() < 0.40:
            defect_type = prev_defect
        else:
            defect_type = rng.choices(DEFECT_TYPES, weights=[50, 10, 14, 9, 8, 5], k=1)[0]
    elif measurement > 60:
        defect_type = rng.choices(DEFECT_TYPES, weights=[35, 15, 15, 12, 12, 11], k=1)[0]
    else:
        defect_type = rng.choices(DEFECT_TYPES, weights=[60, 10, 12, 7, 6, 5], k=1)[0]

    profile = DEFECT_PROFILES[defect_type]
    confidence = round(rng.uniform(*profile["conf_range"]), 1)
    m_status = "within" if defect_type == "None" else rng.choice(["outside", "borderline"])
    analysis_time = round(rng.uniform(1.2, 3.8), 1)

    pname = batch_data["product_name"]
    recurrence = ""
    if defect_type != "None" and prev_defect == defect_type:
        recurrence = f" ⚠️ <b>Recurring defect</b> — same type ({_safe(defect_type)}) found previously."

    if defect_type == "None":
        summary = (
            f"All quality parameters for <b>{_safe(pname)}</b> (Batch: {_safe(batch_id)}) are "
            f"within specifications. No defects detected — <b>{confidence}%</b> confidence. "
            f"Measurement: <b>{measurement}</b> ({m_status} tolerance)."
        )
    else:
        summary = (
            f"<b>{_safe(defect_type)}</b> defect detected on <b>{_safe(pname)}</b> "
            f"(Batch: {_safe(batch_id)}) — <b>{confidence}%</b> confidence. "
            f"Risk: <b>{profile['risk']}</b>. Measurement: <b>{measurement}</b> "
            f"({m_status} tolerance).{recurrence}"
        )

    return {
        "defect_type": defect_type,
        "risk_level": profile["risk"],
        "risk_icon": profile["icon"],
        "risk_color": profile["color"],
        "confidence": confidence,
        "measurement": measurement,
        "measurement_status": m_status,
        "explanation": profile["explanation"],
        "action": profile["action"],
        "action_class": profile["action_class"],
        "image_analyzed": has_image,
        "analysis_time": analysis_time,
        "models_used": ["DefectNet-v3", "SurfaceAnalyzer-v2", "DimensionCheck-v1"],
        "prev_defect": prev_defect,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 4: REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════
# Future: Store reports in Databricks Delta Lake / Azure SQL
# Future: Auto-push to SAP QM module for compliance
# ══════════════════════════════════════════════════════════════════

def generate_report(batch_id: str, batch_data: dict, analysis: dict) -> pd.DataFrame:
    """Generate a structured inspection report DataFrame."""
    report_id = f"QI-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    now = datetime.now()
    strip_html = lambda s: s.replace("<b>", "").replace("</b>", "").replace("&amp;", "&")

    return pd.DataFrame({
        "Field": [
            "Report ID", "Inspection Date", "Inspection Time", "Batch ID",
            "Product Name", "Production Line", "Shift", "Inspector",
            "Previous Defect", "Detected Defect", "AI Confidence", "Risk Level",
            "Measurement", "Measurement Status", "Analysis Duration",
            "AI Models Used", "Image Analyzed", "Summary", "Recommended Action",
        ],
        "Value": [
            report_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), batch_id,
            batch_data["product_name"], batch_data["line"], batch_data["shift"],
            batch_data["inspector"], batch_data.get("prev_defect", "N/A"),
            analysis["defect_type"], f"{analysis['confidence']}%", analysis["risk_level"],
            str(analysis["measurement"]), analysis["measurement_status"],
            f"{analysis['analysis_time']}s", ", ".join(analysis["models_used"]),
            "Yes" if analysis["image_analyzed"] else "No",
            strip_html(analysis["summary"]), strip_html(analysis["action"]),
        ],
    })


def generate_csv_string(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False, quoting=csv.QUOTE_ALL)
    return buf.getvalue()


def generate_txt_report(batch_id: str, batch_data: dict, analysis: dict) -> str:
    sep = "=" * 60
    now = datetime.now()
    strip_html = lambda s: s.replace("<b>", "").replace("</b>", "").replace("&amp;", "&")
    return (
        f"QUALITY INSPECTION REPORT\n{sep}\n"
        f"Report ID   : QI-{now.strftime('%Y%m%d%H%M%S')}\n"
        f"Date        : {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{sep}\n\nBATCH\n{'-'*40}\n"
        f"Batch ID    : {batch_id}\n"
        f"Product     : {batch_data['product_name']}\n"
        f"Line        : {batch_data['line']}\n"
        f"Shift       : {batch_data['shift']}\n"
        f"Inspector   : {batch_data['inspector']}\n\n"
        f"AI ANALYSIS\n{'-'*40}\n"
        f"Defect      : {analysis['defect_type']}\n"
        f"Confidence  : {analysis['confidence']}%\n"
        f"Risk        : {analysis['risk_level']}\n"
        f"Measurement : {analysis['measurement']} ({analysis['measurement_status']})\n\n"
        f"SUMMARY\n{'-'*40}\n{strip_html(analysis['summary'])}\n\n"
        f"ACTION\n{'-'*40}\n{strip_html(analysis['action'])}\n\n"
        f"{sep}\nSmart Quality Inspection System v4.0 — Internal Use Only\n"
    )


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 5: AUDIT TRAIL (DIDI FIX: Issue #07)
# ══════════════════════════════════════════════════════════════════
# Future: Replace with Delta Lake append
#   spark.createDataFrame([row]).write.format("delta").mode("append").save(...)
# ══════════════════════════════════════════════════════════════════

def save_inspection_result(batch_id: str, batch_data: dict, analysis: dict):
    """
    Persist inspection result to CSV file and session_state.

    Future: Delta Lake / Azure SQL write here.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_id": batch_id,
        "product_name": batch_data["product_name"],
        "inspector": batch_data["inspector"],
        "defect_type": analysis["defect_type"],
        "risk_level": analysis["risk_level"],
        "confidence": analysis["confidence"],
        "measurement": analysis["measurement"],
        "measurement_status": analysis["measurement_status"],
    }

    # ── Append to session history ──
    if "inspection_history" not in st.session_state:
        st.session_state["inspection_history"] = []
    st.session_state["inspection_history"].insert(0, row)

    # ── Append to CSV file ──
    try:
        file_exists = os.path.isfile(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass  # Fail silently — file write is non-critical for UI


def get_session_history() -> list:
    return st.session_state.get("inspection_history", [])


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 6: SESSION STATE MANAGEMENT (DIDI FIX: Issue #01)
# ══════════════════════════════════════════════════════════════════

def _on_batch_id_change():
    """
    Callback for Batch ID input change.
    Clears stale analysis results to prevent data mismatch.
    DIDI FIX (Issue #01): Critical — stale results after Batch ID change.
    """
    for key in ["analysis", "batch_data", "active_batch_id", "report_df"]:
        st.session_state.pop(key, None)


def _init_session():
    """Initialize session state defaults."""
    if "inspection_history" not in st.session_state:
        st.session_state["inspection_history"] = []


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 7: CSS THEME — Bosch Enterprise Dark
# ══════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        :root {
            --r: #E20015; --r-dk: #B8000F; --r-lt: #FF1A30;
            --bg1: #0D0D1A; --bg2: #13132B; --bg3: #1A1A35;
            --sf: #1E1E3A; --sf2: #25254A;
            --bd: rgba(255,255,255,0.06);
            --tx: #E4E4F0; --tx2: #9898B4;
            --grn: #00E676; --ylw: #FFD54F; --red: #FF5252;
            --rad: 14px; --shd: 0 8px 32px rgba(0,0,0,0.4);
        }
        .stApp {
            background: linear-gradient(160deg, var(--bg1) 0%, var(--bg2) 40%, var(--bg3) 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--tx);
        }
        #MainMenu, footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111128, #0C0C20);
            border-right: 1px solid rgba(226,0,21,0.1);
        }
        .hero {
            background: linear-gradient(135deg, var(--r), var(--r-dk) 50%, #7A000C);
            padding: 1.6rem 2.5rem; border-radius: var(--rad);
            margin-bottom: 1.4rem; box-shadow: 0 12px 48px rgba(226,0,21,0.25);
            position: relative; overflow: hidden;
        }
        .hero::before {
            content:''; position:absolute; top:-60%; right:-15%;
            width:450px; height:450px;
            background:radial-gradient(circle, rgba(255,255,255,0.07), transparent 70%);
            border-radius:50%;
        }
        .hero .logo {
            font-size:1.2rem; font-weight:900; color:#fff; letter-spacing:4px;
            border:2px solid rgba(255,255,255,0.45); padding:0.2rem 0.8rem;
            border-radius:5px; display:inline-block; margin-bottom:0.4rem;
            position:relative; z-index:1;
        }
        .hero h1 { color:#fff; font-size:1.65rem; font-weight:800; margin:0; position:relative; z-index:1; }
        .hero .sub { color:rgba(255,255,255,0.8); font-size:0.88rem; margin:0.25rem 0 0; position:relative; z-index:1; }

        .kc {
            background:var(--sf); border:1px solid var(--bd); border-radius:var(--rad);
            padding:1rem 1.2rem; text-align:center; box-shadow:var(--shd);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .kc:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(0,0,0,0.5); }
        .kv { font-size:1.9rem; font-weight:800; line-height:1.2; }
        .kl { font-size:0.7rem; color:var(--tx2); text-transform:uppercase; letter-spacing:1.2px; margin-top:0.25rem; }
        .kd { font-size:0.68rem; margin-top:0.15rem; font-weight:600; }

        .sc {
            background:var(--sf); border:1px solid var(--bd); border-radius:var(--rad);
            padding:1.4rem 1.6rem; margin-bottom:0.8rem; box-shadow:var(--shd);
        }
        .sc h3 {
            color:#fff; font-weight:700; font-size:1.05rem; margin:0 0 0.9rem;
            padding-bottom:0.6rem; border-bottom:2px solid var(--r);
        }

        .ag { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:0.7rem; }
        .ai { background:var(--sf2); border:1px solid var(--bd); border-radius:10px; padding:0.7rem 0.9rem; }
        .ai .al { font-size:0.68rem; color:var(--tx2); text-transform:uppercase; letter-spacing:1px; margin-bottom:0.15rem; }
        .ai .av { font-size:0.9rem; font-weight:600; color:#fff; }

        .rp { display:inline-flex; align-items:center; gap:0.3rem; padding:0.4rem 1rem; border-radius:50px; font-weight:700; font-size:0.85rem; }
        .rp-low { background:rgba(0,230,118,0.12); color:var(--grn); border:1px solid rgba(0,230,118,0.25); }
        .rp-medium { background:rgba(255,213,79,0.12); color:var(--ylw); border:1px solid rgba(255,213,79,0.25); }
        .rp-high { background:rgba(255,82,82,0.12); color:var(--red); border:1px solid rgba(255,82,82,0.25); }

        .ib { border-radius:12px; padding:1.1rem 1.3rem; margin:0.5rem 0; line-height:1.7; font-size:0.9rem; }
        .ib.exp { background:linear-gradient(135deg, rgba(226,0,21,0.06), rgba(226,0,21,0.02)); border-left:4px solid var(--r); }
        .ib.suc { background:linear-gradient(135deg, rgba(0,230,118,0.07), rgba(0,230,118,0.02)); border-left:4px solid var(--grn); }
        .ib.wrn { background:linear-gradient(135deg, rgba(255,213,79,0.07), rgba(255,213,79,0.02)); border-left:4px solid var(--ylw); }
        .ib.dgr { background:linear-gradient(135deg, rgba(255,82,82,0.07), rgba(255,82,82,0.02)); border-left:4px solid var(--red); }
        .ib b { color:var(--r); }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stTextArea > div > div > textarea {
            background:var(--sf2)!important; border:1px solid rgba(255,255,255,0.08)!important;
            color:var(--tx)!important; border-radius:10px!important; font-family:'Inter',sans-serif!important;
        }
        .stTextInput > div > div > input:focus {
            border-color:var(--r)!important; box-shadow:0 0 0 2px rgba(226,0,21,0.15)!important;
        }

        div.stButton > button {
            background:linear-gradient(135deg, var(--r), var(--r-dk));
            color:#fff; border:none; border-radius:10px; padding:0.6rem 1.8rem;
            font-weight:700; font-size:0.95rem; transition:all 0.25s;
            box-shadow:0 4px 20px rgba(226,0,21,0.3); font-family:'Inter',sans-serif;
        }
        div.stButton > button:hover {
            transform:translateY(-2px); box-shadow:0 8px 30px rgba(226,0,21,0.45);
        }
        div.stDownloadButton > button {
            background:var(--sf2); color:var(--tx); border:1px solid rgba(226,0,21,0.25);
            border-radius:10px; font-weight:600; font-family:'Inter',sans-serif;
        }
        div.stDownloadButton > button:hover {
            border-color:var(--r); box-shadow:0 4px 15px rgba(226,0,21,0.2);
        }

        .dv { height:1px; background:linear-gradient(90deg, transparent, rgba(226,0,21,0.2), transparent); margin:1.2rem 0; }

        .sb { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:0.9rem 1rem; margin-bottom:0.7rem; }
        .sb h4 { color:#fff; font-size:0.85rem; font-weight:700; margin:0 0 0.4rem; }
        .sb p, .sb li { color:var(--tx2); font-size:0.8rem; line-height:1.6; }
        .sb ol, .sb ul { padding-left:1rem; }

        .sd { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; animation:pulse 2s infinite; }
        .sd.on { background:var(--grn); }
        @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

        .ft { text-align:center; color:var(--tx2); font-size:0.72rem; padding:1.5rem 0 0.5rem; border-top:1px solid var(--bd); margin-top:1.5rem; }

        .tb { display:flex; align-items:flex-end; gap:3px; height:45px; }
        .tr { flex:1; min-width:8px; background:linear-gradient(180deg, var(--r), rgba(226,0,21,0.3)); border-radius:3px 3px 0 0; }

        .img-preview {
            border: 2px solid rgba(226,0,21,0.3); border-radius: 10px;
            padding: 0.5rem; background: var(--sf2); text-align: center;
        }
        .img-preview img { max-height: 150px; border-radius: 6px; }
        .img-badge {
            display:inline-block; margin-top:0.4rem; padding:0.25rem 0.8rem;
            background:rgba(226,0,21,0.1); border:1px solid rgba(226,0,21,0.2);
            border-radius:20px; font-size:0.72rem; color:var(--ylw); font-weight:600;
        }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 8: UI RENDERING
# ══════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:0.6rem 0 0.2rem;">
            <div style="font-size:1.2rem; font-weight:900; color:#E20015; letter-spacing:4px;
                        border:2px solid rgba(226,0,21,0.35); display:inline-block;
                        padding:0.2rem 0.8rem; border-radius:5px;">BOSCH</div>
            <p style="color:#7777A0; font-size:0.68rem; margin-top:0.3rem; letter-spacing:1.5px;">
                QUALITY ENGINEERING DIVISION</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

        d_status = "Loaded" if not df.empty else "Missing"
        d_color = "#00E676" if not df.empty else "#FF5252"
        sess = get_session_history()
        st.markdown(f"""
        <div class="sb">
            <h4>📡 System Status</h4>
            <p style="margin:0 0 0.2rem;">
                <span class="sd on"></span>
                <span style="color:#00E676; font-weight:600;">Online</span>
                <span style="color:#555; font-size:0.75rem;"> — All modules active</span>
            </p>
            <p style="margin:0 0 0.2rem;">
                <span style="color:{d_color}; font-weight:600;">📊 Dataset:</span>
                <span style="color:{d_color};">{d_status} ({len(df)} records)</span>
            </p>
            <p style="margin:0;">
                <span style="color:#00E676; font-weight:600;">🔬 Session:</span>
                <span style="color:#00E676;">{len(sess)} inspections completed</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sb">
            <h4>⚡ How It Works</h4>
            <ol>
                <li>Enter a <b>Batch ID</b></li>
                <li>System <b>auto-fills</b> from CSV</li>
                <li>AI <b>auto-analyzes</b> instantly</li>
                <li>Report <b>auto-generated</b></li>
                <li>Result <b>auto-saved</b> to audit trail</li>
            </ol>
            <p style="color:#00E676; font-size:0.78rem; font-weight:600; margin-top:0.3rem;">
                ✨ Zero manual steps — fully automated flow</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sb">
            <h4>📉 Manual Effort Eliminated</h4>
            <p>
                <b style="color:#00E676;">✓</b> No defect selection<br>
                <b style="color:#00E676;">✓</b> No form filling<br>
                <b style="color:#00E676;">✓</b> No report writing<br>
                <b style="color:#00E676;">✓</b> No manual saving<br>
                <b style="color:#00E676;">✓</b> Auto-audit trail
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sb">
            <h4>🔮 Upcoming Integrations</h4>
            <ul>
                <li><b>Databricks ML</b> — Live prediction</li>
                <li><b>Computer Vision</b> — Image analysis</li>
                <li><b>Azure IoT Hub</b> — Sensor stream</li>
                <li><b>SAP ERP</b> — Order sync</li>
                <li><b>Delta Lake</b> — Audit storage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if not df.empty:
            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            ids = df["batch_id"].tolist()
            sample = "<br>".join(ids[:8])
            extra = f'<br><span style="color:#666; font-size:0.7rem;">+ {len(ids)-8} more</span>' if len(ids) > 8 else ""
            st.markdown(f"""
            <div class="sb">
                <h4>🏷️ Batch IDs in Dataset</h4>
                <p style="font-family:monospace; font-size:0.78rem; line-height:1.8;">{sample}{extra}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; padding:0.4rem 0;">
            <p style="color:#555; font-size:0.68rem; margin:0;">
                v4.0.0 — DIDI-Refactored<br>{datetime.now().strftime("%B %d, %Y")}<br>
                <span style="color:#E20015;">●</span> Internal Use Only</p>
        </div>
        """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="hero">
        <div class="logo">BOSCH</div>
        <h1>🔬 Smart Quality Inspection System</h1>
        <div class="sub">AI-Powered · Fully Automated · CSV-Driven · Audit-Ready</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(kpis: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kc">
            <div class="kv" style="color:#fff;">{kpis['sess_total']}</div>
            <div class="kl">Inspections (Session)</div>
            <div class="kd" style="color:var(--tx2);">📊 {kpis['total_csv']} in dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pc = kpis["csv_pass_rate"]
        pc_c = "#00E676" if pc >= 95 else "#FFD54F" if pc >= 85 else "#FF5252"
        st.markdown(f"""
        <div class="kc">
            <div class="kv" style="color:{pc_c};">{pc}%</div>
            <div class="kl">Historical Pass Rate</div>
            <div class="kd" style="color:{pc_c};">{'▲ Healthy' if pc >= 90 else '⚠ Needs attention'}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        sd = kpis["sess_defects"]
        sd_c = "#00E676" if sd == 0 else "#FFD54F" if sd < 3 else "#FF5252"
        st.markdown(f"""
        <div class="kc">
            <div class="kv" style="color:{sd_c};">{sd}</div>
            <div class="kl">Defects (Session)</div>
            <div class="kd" style="color:var(--tx2);">{kpis['csv_defects']} historical</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sc = kpis["sess_critical"]
        sc_c = "#FF5252" if sc > 0 else "#00E676"
        st.markdown(f"""
        <div class="kc">
            <div class="kv" style="color:{sc_c};">{sc}</div>
            <div class="kl">Critical Alerts (Session)</div>
            <div class="kd" style="color:{sc_c};">{'🚨 Action needed' if sc > 0 else '✅ All clear'}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


def render_autofill(batch_data: dict, batch_id: str):
    pr = batch_data["prev_pass_rate"]
    pr_c = "#00E676" if pr >= 95 else "#FFD54F" if pr >= 90 else "#FF5252"
    pd_v = batch_data["prev_defect"]
    pd_c = "#00E676" if pd_v == "None" else "#FF5252" if pd_v in ["Crack","Misalignment","Dimensional Deviation"] else "#FFD54F"

    st.markdown(f"""
    <div class="sc">
        <h3>📋 Auto-Filled Data — <span style="color:var(--r);">{_safe(batch_id)}</span>
            <span style="float:right; font-size:0.72rem; color:var(--tx2); font-weight:400;">Source: quality_data.csv</span>
        </h3>
        <div class="ag">
            <div class="ai"><div class="al">📦 Product</div><div class="av">{batch_data['product_name']}</div></div>
            <div class="ai"><div class="al">👤 Inspector</div><div class="av">{batch_data['inspector']}</div></div>
            <div class="ai"><div class="al">🏭 Line</div><div class="av">{batch_data['line']}</div></div>
            <div class="ai"><div class="al">🕐 Shift</div><div class="av">{batch_data['shift']}</div></div>
            <div class="ai"><div class="al">📅 Date</div><div class="av">{batch_data['date']}</div></div>
            <div class="ai"><div class="al">📏 Measurement</div><div class="av">{batch_data['measurement']}</div></div>
            <div class="ai"><div class="al">🔄 Prev Defect</div><div class="av" style="color:{pd_c};">{pd_v}</div></div>
            <div class="ai"><div class="al">📊 Prev Inspections</div><div class="av">{batch_data['prev_inspections']}</div></div>
            <div class="ai"><div class="al">✅ Pass Rate</div><div class="av" style="color:{pr_c};">{pr}%</div></div>
            <div class="ai" style="grid-column:span 2;"><div class="al">📝 Remarks</div><div class="av" style="font-size:0.82rem; font-weight:400;">{batch_data['remarks']}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_image_preview(uploaded_file):
    """DIDI FIX (Issue #03): Show uploaded image with status badge."""
    if uploaded_file is not None:
        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 3])
        with col_img:
            st.image(uploaded_file, caption=None, width=180)
        with col_info:
            st.markdown(f"""
            <div class="sc" style="padding:1rem;">
                <h3 style="font-size:0.9rem;">📷 Uploaded Image</h3>
                <div class="ai" style="margin-bottom:0.4rem;">
                    <div class="al">File Name</div>
                    <div class="av">{_safe(uploaded_file.name)}</div>
                </div>
                <div class="ai" style="margin-bottom:0.4rem;">
                    <div class="al">File Size</div>
                    <div class="av">{round(uploaded_file.size / 1024, 1)} KB</div>
                </div>
                <div class="img-badge">🔬 Queued for AI Vision Analysis</div>
                <p style="font-size:0.75rem; color:var(--tx2); margin-top:0.4rem;">
                    Image will be processed when Computer Vision model is integrated.</p>
            </div>
            """, unsafe_allow_html=True)


def render_analysis(analysis: dict, batch_data: dict, df: pd.DataFrame):
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sc"><h3>🤖 AI Analysis Dashboard</h3></div>', unsafe_allow_html=True)

    risk = analysis["risk_level"]
    rc = risk.lower()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kc">
            <div class="kl" style="margin-bottom:0.4rem;">Detected Defect</div>
            <div class="kv" style="font-size:1.2rem; color:{analysis['risk_color']};">{_safe(analysis['defect_type'])}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kc">
            <div class="kl" style="margin-bottom:0.4rem;">Risk Level</div>
            <div class="rp rp-{rc}" style="margin-top:0.2rem;">{analysis['risk_icon']} {risk}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kc">
            <div class="kl" style="margin-bottom:0.4rem;">AI Confidence</div>
            <div class="kv" style="font-size:1.4rem; color:{analysis['risk_color']};">{analysis['confidence']}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kc">
            <div class="kl" style="margin-bottom:0.4rem;">Analysis Time</div>
            <div class="kv" style="font-size:1.4rem;">{analysis['analysis_time']}s</div>
            <div style="font-size:0.68rem; color:var(--tx2); margin-top:0.15rem;">{len(analysis['models_used'])} models active</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(f'<div class="ib exp"><b>📊 Summary</b><br><br>{analysis["summary"]}</div>', unsafe_allow_html=True)

    col_d, col_h = st.columns([3, 2])
    with col_d:
        st.markdown(f'<div class="ib exp"><b>🔎 Detailed Analysis</b><br><br>{analysis["explanation"]}</div>', unsafe_allow_html=True)
        ac = {"success": "suc", "warning": "wrn", "danger": "dgr"}[analysis["action_class"]]
        st.markdown(f'<div class="ib {ac}"><b style="color:{analysis["risk_color"]};">🎯 Recommended Action</b><br><br>{analysis["action"]}</div>', unsafe_allow_html=True)

    with col_h:
        prev = analysis.get("prev_defect", "N/A")
        recur = ""
        if prev != "None" and prev == analysis["defect_type"]:
            recur = '<div style="margin-top:0.4rem; padding:0.3rem 0.7rem; background:rgba(255,82,82,0.1); border:1px solid rgba(255,82,82,0.2); border-radius:8px; font-size:0.78rem; color:#FF5252; font-weight:600;">⚠️ Recurring Defect Pattern</div>'

        tags = "".join(f'<span style="display:inline-block; background:var(--sf2); border:1px solid var(--bd); border-radius:6px; padding:0.15rem 0.5rem; font-size:0.7rem; color:var(--tx); margin:0.1rem 0.15rem 0.1rem 0;">{m}</span>' for m in analysis["models_used"])

        # Defect trend from CSV
        trend_html = ""
        if not df.empty:
            ddf = df[df["previous_defect"].str.strip().str.lower() != "none"]
            if not ddf.empty:
                counts = ddf.groupby("date").size().sort_index()
                mx = counts.max() if counts.max() > 0 else 1
                bars = "".join(f'<div class="tr" style="height:{max(int((c/mx)*40),4)}px;" title="{d}: {c}"></div>' for d, c in counts.items())
                dl = list(counts.index)
                trend_html = f"""
                <div style="margin-top:0.4rem;">
                    <div style="font-size:0.68rem; color:var(--tx2); margin-bottom:0.2rem; letter-spacing:0.5px;">DEFECTS BY DATE (CSV)</div>
                    <div class="tb">{bars}</div>
                    <div style="display:flex; justify-content:space-between; font-size:0.58rem; color:#555; margin-top:0.15rem;">
                        <span>{dl[0]}</span><span>{dl[-1]}</span></div>
                </div>"""

        st.markdown(f"""
        <div class="sc" style="padding:1.1rem 1.3rem;">
            <h3 style="font-size:0.9rem;">📈 Historical Context</h3>
            <div class="ai" style="margin-bottom:0.4rem;"><div class="al">Batch History</div><div class="av">{batch_data['prev_inspections']} prior</div></div>
            <div class="ai" style="margin-bottom:0.4rem;"><div class="al">Pass Rate</div><div class="av" style="color:{'#00E676' if batch_data['prev_pass_rate']>=95 else '#FFD54F'};">{batch_data['prev_pass_rate']}%</div></div>
            <div class="ai" style="margin-bottom:0.4rem;"><div class="al">Previous Defect</div><div class="av">{_safe(prev)}</div></div>
            <div class="ai" style="margin-bottom:0.4rem;"><div class="al">Current Result</div><div class="av" style="color:{analysis['risk_color']};">{_safe(analysis['defect_type'])} — {risk}</div></div>
            <div class="ai" style="margin-bottom:0.4rem;"><div class="al">Image</div><div class="av">{'Yes ✅' if analysis['image_analyzed'] else 'No — data-based'}</div></div>
            {recur}{trend_html}
            <div style="margin-top:0.6rem;">
                <div style="font-size:0.68rem; color:var(--tx2); letter-spacing:0.5px; margin-bottom:0.2rem;">AI MODELS</div>
                {tags}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_report(batch_id: str, batch_data: dict, analysis: dict):
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sc"><h3>📄 Auto-Generated Report</h3></div>', unsafe_allow_html=True)

    report_df = generate_report(batch_id, batch_data, analysis)
    csv_data = generate_csv_string(report_df)
    txt_data = generate_txt_report(batch_id, batch_data, analysis)
    report_id = report_df[report_df["Field"] == "Report ID"]["Value"].values[0]

    st.dataframe(report_df, width="stretch", hide_index=True, height=680)

    st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.download_button("📊 Download CSV Report", csv_data, f"report_{report_id}.csv", "text/csv")
    with c2:
        st.download_button("📝 Download TXT Summary", txt_data, f"summary_{report_id}.txt", "text/plain")


def render_recent_inspections():
    """DIDI FIX (Issue #07): Display audit trail of session inspections."""
    history = get_session_history()
    if not history:
        return

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sc"><h3>📜 Recent Inspections — Audit Trail</h3></div>', unsafe_allow_html=True)

    hist_df = pd.DataFrame(history[:20])
    display_cols = ["timestamp", "batch_id", "product_name", "defect_type", "risk_level", "confidence"]
    available = [c for c in display_cols if c in hist_df.columns]

    if available:
        styled = hist_df[available].copy()
        styled.columns = ["Time", "Batch", "Product", "Defect", "Risk", "Conf%"][:len(available)]
        st.dataframe(styled, width="stretch", hide_index=True, height=min(len(styled) * 40 + 60, 350))

    if os.path.isfile(RESULTS_CSV):
        st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)
        try:
            with open(RESULTS_CSV, "r", encoding="utf-8") as f:
                audit_csv = f.read()
            st.download_button("📋 Download Full Audit Trail", audit_csv, "audit_trail.csv", "text/csv")
        except Exception:
            pass


def render_footer():
    st.markdown(f"""
    <div class="ft">
        <b>Smart Quality Inspection System</b> — v4.0.0 DIDI-Refactored<br>
        © {datetime.now().year} Engineering Division — Internal Use Only<br>
        <span style="color:#E20015;">Invented for life.</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ██  MODULE 9: MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════

def main():
    inject_css()
    _init_session()

    # ── Load data ──
    df = load_data()
    kpis = compute_kpis(df, get_session_history())

    # ── Sidebar ──
    render_sidebar(df)

    # ── Header + KPIs ──
    render_header()
    render_kpis(kpis)

    # ── Input Section — DIDI FIX: removed voice placeholder ──
    st.markdown('<div class="sc"><h3>⚡ Batch Inspection — Enter ID to Auto-Analyze</h3></div>', unsafe_allow_html=True)

    col_input, col_upload = st.columns([2, 1.5])
    with col_input:
        batch_id = st.text_input(
            "🏷️ Batch ID",
            placeholder="e.g., BX-2026-04-0892",
            help="Enter the batch ID. Everything else is automatic.",
            key="batch_input",
            on_change=_on_batch_id_change,  # DIDI FIX: clear stale state
        )
    with col_upload:
        uploaded_file = st.file_uploader(
            "📷 Inspection Image (Optional)",
            type=["png", "jpg", "jpeg"],
            help="Upload a component photo. Will be queued for AI vision analysis.",
        )

    # ── DIDI FIX (Issue #03): Image preview ──
    if uploaded_file is not None:
        render_image_preview(uploaded_file)

    # ── Auto-analyze flow (DIDI FIX: Issue #01, #02 — no button needed) ──
    bid = batch_id.strip() if batch_id else ""
    if bid:
        batch_data = fetch_batch_data(df, bid)
        if batch_data:
            render_autofill(batch_data, bid)

            # ── Auto-trigger analysis (DIDI FIX: removed Analyze button) ──
            current_key = f"{bid}:{datetime.now().strftime('%Y%m%d%H%M')}"
            if st.session_state.get("last_analysis_key") != current_key:
                with st.spinner("🧠 Running AI defect detection..."):
                    analysis = simulate_ai_analysis(bid, batch_data, uploaded_file is not None)

                # Future: Databricks ML Serving API call replaces above

                st.session_state["analysis"] = analysis
                st.session_state["batch_data"] = batch_data
                st.session_state["active_batch_id"] = bid
                st.session_state["last_analysis_key"] = current_key

                # DIDI FIX (Issue #07): Auto-save to audit trail
                save_inspection_result(bid, batch_data, analysis)

            # ── Display results ──
            if "analysis" in st.session_state and st.session_state.get("active_batch_id") == bid:
                render_analysis(st.session_state["analysis"], st.session_state["batch_data"], df)
                render_report(bid, st.session_state["batch_data"], st.session_state["analysis"])

        else:
            st.warning(f"⚠️ Batch ID **`{_safe(bid)}`** not found in dataset. Check the sidebar for valid IDs.")
            # Clear any stale results
            for key in ["analysis", "batch_data", "active_batch_id"]:
                st.session_state.pop(key, None)

    # ── Audit Trail — always visible ──
    render_recent_inspections()

    # ── Footer ──
    render_footer()


if __name__ == "__main__":
    main()
