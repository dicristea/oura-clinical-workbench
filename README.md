# 🏥 Clinical Coordinator Dashboard

> **Oura Ring Study Data Monitoring Platform**

A web-based dashboard for clinical coordinators to monitor patient participation in the Oura Ring biometric study, track data collection status, and analyze patient biometrics with interactive charts.

---

## 🌐 Live Demo

👉 **[View Live Dashboard](https://oura-clinical-dashboard.onrender.com)**

---

## 📸 Dashboard Preview

![Clinical Coordinator Dashboard](https://raw.githubusercontent.com/AlbinaKrasykova/oura-clinical-dashboard/main/view1.png)
![Clinical Coordinator Dashboard](https://raw.githubusercontent.com/AlbinaKrasykova/oura-clinical-dashboard/main/view2.png)
![Clinical Coordinator Dashboard](https://raw.githubusercontent.com/AlbinaKrasykova/oura-clinical-dashboard/main/view2.2.png)


---

## ✨ Features

### Main Dashboard
- **Patient Overview Table** - View all patients at a glance
- **Data Collection Status** - Track inpatient/outpatient data counts
- **Sync Monitoring** - See when data was last synced
- **Status Indicators** - Oura ✓ and EHR ✓ connection status
- **Smart Alerts** - Active, Needs Follow-up, Outreach Needed badges
- **3 Metric Trends** - Sleep Score, HRV Average, Activity Score sparklines
- **Filtering** - Filter by status, data overlap, completion
- **Search** - Search patients by ID

### Patient Detail View (Click on Patient ID)
- **Combined View** - All 3 metrics on one line chart
  - 🔵 HRV Average (blue)
  - 🟠 Activity Score (orange)  
  - 🔵 Sleep Score (cyan)
- **Separate Views** - Individual dot charts for each metric
- **Interactive Timeline Slider**
  - Drag left handle to expand/shrink from left
  - Drag right handle to expand/shrink from right
  - Drag middle to move the window
- **Real-time Updates** - Charts update as you drag

---

## 🎯 Purpose

This dashboard helps clinical research coordinators:

| Task | How Dashboard Helps |
|------|---------------------|
| **Monitor participation** | See all patients and their status at a glance |
| **Identify issues** | Color-coded badges highlight who needs attention |
| **Track data quality** | View Oura + EHR data overlap |
| **Analyze trends** | Interactive charts show patient biometrics over time |
| **Prioritize outreach** | Filter to see only patients needing follow-up |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Data:** Pandas, Excel (xlsx)
- **Charts:** Custom SVG charts with interactive timeline
- **Hosting:** Render (free tier)

---

## 📁 Project Structure

```
oura-clinical-dashboard/
├── app.py                    # Flask application
├── templates/
│   ├── dashboard.html        # Main dashboard view
│   └── patient_detail.html   # Patient metrics detail view
├── demo_data.xlsx            # Sample patient data
├── requirements_deploy.txt   # Python dependencies
├── render.yaml               # Render deployment config
├── README.md                 # This file
│
└── HF-Notebook/              # Original Jupyter analysis notebooks
    ├── config.py             # Configuration loader
    ├── vis.py                # Visualization helpers
    └── requirements.txt      # Notebook dependencies
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

## 📊 Dashboard Views

### Main Dashboard
| Column | Description |
|--------|-------------|
| **Patient ID** | Click to view detailed metrics |
| **Data Collection** | Inpatient/outpatient counts + sync status |
| **Metric Trends** | Sleep, HRV, Activity sparklines |
| **Participation Dates** | Study enrollment period |
| **Hospitalization Dates** | Hospital admission period |

### Patient Detail View
| View | Description |
|------|-------------|
| **Combined View** | All metrics overlaid on single chart |
| **Separate Views** | Individual charts for each metric |
| **Timeline Slider** | Interactive date range selector |

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
