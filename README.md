# 🔬 AI-Powered Smart Quality Inspection System

> Enterprise-grade, semi-automated quality inspection frontend for manufacturing environments. Designed for Bosch-style industrial workflows.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-Internal-orange)

## ✨ Features

- **Zero Manual Entry** — Enter a Batch ID and everything auto-fills from CSV dataset
- **Auto AI Analysis** — Defect detection runs automatically, no button clicks needed
- **Audit Trail** — Every inspection auto-saved to `results/inspections.csv`
- **Session KPIs** — Live dashboard tracking inspections, defects, and critical alerts
- **Image Upload** — Queued for future Computer Vision integration
- **Downloadable Reports** — CSV and TXT report generation
- **Bosch Dark Theme** — Professional enterprise UI with red accent design system

## 🚀 Quick Start

```bash
# Install dependencies
pip install streamlit pandas

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
quality-inspection/
├── app.py                 # Main application (9 modular sections)
├── quality_data.csv       # Inspection dataset (20 batch records)
├── results/               # Auto-generated audit trail directory
│   └── inspections.csv    # Persistent inspection log
├── .gitignore
└── README.md
```

## 🏗️ Architecture

The app is organized into 9 clearly separated modules (single-file for portability):

| Module | Purpose |
|--------|---------|
| Configuration | Paths, constants, page config |
| Data Layer | CSV loading, batch lookup, KPI computation |
| AI Engine | Simulated defect detection with weighted logic |
| Report Generator | DataFrame + CSV/TXT report creation |
| Audit Trail | Session history + CSV persistence |
| State Management | Session state lifecycle, stale-data prevention |
| CSS Theme | Bosch enterprise dark theme |
| UI Rendering | All render functions (sidebar, KPIs, panels) |
| Main | Orchestration and flow control |

## 🔮 Future Integration Points

The code contains clearly marked integration comments:

- `# Future: Databricks Delta Lake query` — Data layer
- `# Future: Databricks ML Serving API` — AI engine
- `# Future: Computer Vision model` — Image analysis
- `# Future: SAP ERP REST API` — Batch data fetch
- `# Future: Delta Lake write` — Audit persistence

## ⚙️ Requirements

- Python 3.10+
- Streamlit
- Pandas

## 📊 Dataset

`quality_data.csv` contains 20 sample batch records with fields:
- Batch ID, Product Name, Inspector, Production Line
- Shift, Date, Measurement Value
- Previous Defect, Previous Inspections, Pass Rate, Remarks

---

**Internal Use Only** · v4.0.0 · DIDI-Refactored
