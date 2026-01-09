# Clinical Coordinator Dashboard

> **Oura Ring Study Data Monitoring Platform**

A web-based dashboard for clinical coordinators to monitor patient participation in the Oura Ring biometric study, track data collection status, and identify patients needing follow-up.

---

## Live Demo

👉 **[View Live Dashboard](https://oura-clinical-dashboard.onrender.com)**

---

## Dashboard Preview

![Clinical Coordinator Dashboard](https://github.com/AlbinaKrasykova/oura-clinical-dashboard/blob/main/mockup.png?raw=true)

### Features Shown:
- **Patient Overview Table** - View all patients at a glance
- **Data Collection Status** - Track inpatient/outpatient data counts
- **Sync Monitoring** - See when data was last synced
- **Status Indicators** - Oura ✓ and EHR ✓ connection status
- **Smart Alerts** - Active, Needs Follow-up, Outreach Needed badges
- **Metric Trends** - Sparkline charts for each patient
- **Filtering** - Filter by status, data overlap, completion

---

## 🎯 Purpose

This dashboard helps clinical research coordinators:

| Task | How Dashboard Helps |
|------|---------------------|
| **Monitor participation** | See all patients and their status at a glance |
| **Identify issues** | Color-coded badges highlight who needs attention |
| **Track data quality** | View Oura + EHR data overlap |
| **Prioritize outreach** | Filter to see only patients needing follow-up |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Data:** Pandas, Excel (xlsx)
- **Charts:** SVG Sparklines
- **Hosting:** Render (free tier)

---

## 📁 Project Structure

```
oura-clinical-dashboard/
├── app.py                    # Flask application
├── templates/
│   └── dashboard.html        # Dashboard UI (exact mockup design)
├── demo_data.xlsx            # Sample patient data
├── requirements_deploy.txt   # Python dependencies
├── render.yaml               # Render deployment config
├── README.md                 # This file
│
└── HF-Notebook/              # Original Jupyter analysis notebooks
    ├── config.py             # Configuration loader
    ├── vis.py                # Visualization helpers
    ├── requirements.txt      # Notebook dependencies
    └── flowsheet_oura_vis_naomi-Copy1.ipynb
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repo
```bash
git clone https://github.com/AlbinaKrasykova/oura-clinical-dashboard.git
cd oura-clinical-dashboard
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements_deploy.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
👉 http://localhost:5000

---

## 📊 Data Format

The dashboard expects an Excel file with these columns:

| Column | Description |
|--------|-------------|
| `mrn` | Patient Medical Record Number |
| `first_name` | Patient first name |
| `last_name` | Patient last name |
| `token` | Oura API token (indicates Oura connection) |
| `clarity_admit_date` | Hospital admission date |
| `clarity_discharge_date` | Hospital discharge date |
| `flowsheet_entry_datetime` | Last EHR data entry time |

---

## 🔒 Security Notes

⚠️ **Important for PHI/HIPAA Compliance:**

- Real patient data (`data.xlsx`) is **NOT** committed to this repo
- Only `demo_data.xlsx` with fake sample data is included
- Keep `.env` files with API tokens out of version control
- Use environment variables for sensitive configuration

---

## 🙏 Credits

Built as an extension of the [HF-Notebook](https://github.com/TomorrowMC/HF-Notebook) project for Oura Ring biometric data analysis.

---

## 📄 License

This project handles Protected Health Information (PHI). Ensure compliance with:
- HIPAA regulations
- IRB requirements  
- Data use agreements
- Patient consent requirements

---

## 📬 Contact

**Albina Krasykova**  
Cornell Medicine Research

---

*Built with ❤️ for clinical research*
