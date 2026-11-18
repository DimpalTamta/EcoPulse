# 🌿 EcoPulse — Air & Water Intelligence for India

**Live Application:** [https://ecopulse-water-air.streamlit.app/](https://ecopulse-water-air.streamlit.app/)

**GitHub Repository:** [https://github.com/DimpalTamta/EcoPulse](https://github.com/DimpalTamta/EcoPulse)

EcoPulse is an integrated environmental intelligence application that combines **air quality analysis, water quality monitoring, machine learning predictions, and interactive visualisations** for India.

The project uses air quality data covering 26 cities (2015–July 2020) and river water quality data from CPCB monitoring stations (2012–2023).

Built using Python, Pandas, NumPy, Scikit-learn, Plotly, and Streamlit.

## 🚀 Run the Application Locally

### 1. Requirements

- Python 3.11–3.13
- pip

### 2. Clone the repository

```bash
git clone https://github.com/DimpalTamta/EcoPulse.git
cd EcoPulse
```

### 3. Create and activate a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app/main.py
```

**Windows shortcut:** You can also run `run_app.bat` after installing the dependencies.

> **Important:** Keep `scikit-learn==1.5.2` in `requirements.txt` because the saved air machine learning models were trained using this version.

## ☁️ Live Deployment

EcoPulse is deployed using **Streamlit Community Cloud**.

🌐 **[Open EcoPulse — Live Application](https://ecopulse-water-air.streamlit.app/)**

## 📊 Application Features

| Category | Page | Description |
|---|---|---|
| Home | Home Dashboard | Environmental overview, key statistics, air and water charts, and EcoPulse Score |
| Air Quality | Air Dashboard | City and year filters, pollution trends, calendar heatmap, pollutant distribution, and comparisons |
| Air Quality | PM2.5 Prediction | Random Forest-based PM2.5 estimation with a colour gauge and health category |
| Air Quality | Historical Trends | Historical timelines, seasonal patterns, and pollutant correlations |
| Air Quality | Air Analytics | City-year heatmaps, seasonality, AQI category distribution, weekday analysis, and correlations |
| Air Quality | Pollution Clusters & PCA | K-Means clustering, pollution patterns, and PCA visualisation |
| Water Quality | Water Dashboard | State and year filters, water quality classification, parameter distributions, and station comparisons |
| Water Quality | Water Trends & States | Yearly parameter trends, state-level heatmaps, water quality class distribution, and coverage analysis |
| Water Quality | Water Analytics | Correlation analysis, K-Means clustering, PCA, and parameter distribution |
| Combined Analysis | Air & Water Insights | State-wise EcoPulse Score, air-water comparisons, and analytical summaries |
| Combined Analysis | India Pollution Map & Hotspots | Geographic pollution visualisation and hotspot analysis |
| Reports | Air ML Comparison | Comparison of regression models using MAE, RMSE, and R² |
| Reports | Download Centre | Downloadable PDF and CSV reports for air and water analysis |

## 🌊 Water Quality Data Processing

The water quality dataset contains inconsistent records and values outside expected physical ranges.

The preprocessing process includes:

- Removing invalid values using defined physical-range checks.
- Handling missing values using median-based imputation.
- Normalising inconsistent state names.
- Preparing cleaned datasets for analysis and visualisation.
- Applying simplified water quality screening criteria.

**Water quality classification note:** The screening class is based on selected parameters, including dissolved oxygen, pH, BOD, and total coliform. It is an analytical approximation and should not be interpreted as an official CPCB certification.

## 🌱 EcoPulse Score

The EcoPulse Score provides a combined state-level environmental indicator using:

- An air quality score based on PM2.5 concentration.
- The proportion of water quality records meeting at least Class C.

The score is calculated by averaging the air score and the water quality share.

It is an analytical indicator developed for this project, not an official government environmental index.

## 🎨 Colour Interpretation

### Air Quality — PM2.5

| Category | PM2.5 Range (µg/m³) |
|---|---:|
| Good | 0–30 |
| Satisfactory | 31–60 |
| Moderate | 61–90 |
| Poor | 91–120 |
| Very Poor | 121–250 |
| Severe | Above 250 |

### Water Quality

| Colour | Classification |
|---|---|
| Green | Class A |
| Light Green | Class B |
| Amber | Class C |
| Red | Below Class C |

## 📁 Project Structure

```text
EcoPulse/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── main.py
│   ├── theme.py
│   ├── water_data.py
│   └── pages/
│       ├── air_dashboard.py
│       ├── air_prediction.py
│       ├── air_trends.py
│       ├── air_analytics.py
│       ├── air_clusters.py
│       ├── water_dashboard.py
│       ├── water_trends.py
│       ├── water_analytics.py
│       ├── insights.py
│       ├── pollution_map.py
│       ├── pollution_hotspots.py
│       ├── ml_comparison.py
│       ├── report_center.py
│       ├── report_generator.py
│       └── water_report.py
│
├── data/
│   ├── raw/
│   │   ├── air/
│   │   └── water/
│   └── processed/
│       ├── air_clusters.csv
│       ├── air_pca.csv
│       ├── air_pm25.csv
│       └── water/
│
├── models/
│   ├── air_decision_tree.pkl
│   ├── air_kmeans.pkl
│   ├── air_lasso_regression.pkl
│   ├── air_linear_regression.pkl
│   ├── air_pca.pkl
│   ├── air_random_forest.pkl
│   ├── air_ridge_regression.pkl
│   └── water/
│
├── notebooks/
│   └── air_eda.ipynb
│
├── reports/
│   ├── figures/
│   ├── water/
│   ├── air_cluster_summary.csv
│   └── air_model_comparison.csv
│
├── src/
│   ├── preprocess_air.py
│   ├── train_air_models.py
│   ├── evaluate_air_models.py
│   └── other analysis and training scripts
│
├── requirements.txt
├── README.md
├── .gitignore
├── run_app.bat
└── run_app.sh
```

## 🧠 Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualisation:** Plotly, Matplotlib
- **Web Application:** Streamlit
- **Model Storage:** Pickle, Joblib
- **Report Generation:** ReportLab
- **Data Format:** CSV

## ✨ Design and Interface

EcoPulse includes a consistent visual design system with multiple colour palettes:

- EcoPulse Teal
- Ocean Blue
- Forest Green
- Purple Modern
- Clean Professional

The interface includes interactive charts, filters, analytical summaries, responsive layouts, and a custom sidebar.

## ⚠️ Disclaimer

EcoPulse is an academic project developed for educational and analytical purposes.

The results, predictions, classifications, and EcoPulse Score are intended for demonstration and research. They should not be treated as official CPCB assessments or used as a substitute for authoritative environmental monitoring data.

---

**EcoPulse — Making Environmental Data More Understandable.**
