# 📊 DataLens — AI-Powered Exploratory Data Analysis Platform

An interactive web application for exploring, cleaning, and understanding CSV datasets — combining automated statistical analysis with an AI assistant for natural-language data insights.

Upload any CSV and DataLens automatically profiles the data, flags quality issues, visualizes distributions, and lets you ask questions about your dataset in plain English.

---

## 📖 About the Project

Exploratory Data Analysis is usually the slowest part of any data science workflow — inspecting column types, checking for missing values, hunting for outliers, and writing the first summary of what a dataset actually contains.

DataLens automates that first pass. It combines:

- **Rule-based statistical analysis** — missing values, duplicates, correlations, skewness, and outlier detection computed directly from the data
- **An AI assistant powered by Google Gemini** — generates a plain-English summary of the dataset and answers free-form questions about it in a chat interface

The goal is a tool that gets a data analyst from "raw CSV" to "first understanding" in seconds, not an hour of manual `.describe()` calls.

---

## ✨ Features

- **Smart CSV import** — automatic separator and encoding detection
- **Overview dashboard** — row/column counts, missing value %, duplicate detection, live charts
- **Data Explorer** — per-column type, distribution, and summary statistics
- **Visualizations** — interactive Plotly charts across the dataset
- **Data Quality tools** — detect and clean missing values, duplicates, and numerical outliers
- **AI Insights** — one-click AI-generated dataset summary, plus a chat box to ask questions about your data
- **Export** — download a cleaned dataset and a full generated EDA report

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| App Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| AI | Google Gemini API (`google-genai`) |

---

## 🚀 Live Demo

🔗 *[Add your Streamlit Community Cloud link here once deployed]*

---

## 💻 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/deviprasadchebodula/datalens-eda.git
cd datalens-eda
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Create a file at `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**4. Run the app**
```bash
streamlit run DataLens_Pro_v7.py
```

---

## 📁 Project Structure

```
datalens-eda/
├── DataLens_Pro_v7.py     # Main application
├── requirements.txt       # Python dependencies
└── .gitignore              # Excludes local secrets file
```

---

## 🙋‍♂️ Author

**Chebodula Devi Prasad**
B.Tech CSE – Data Science | Data Analyst & ML Engineer

[LinkedIn](https://www.linkedin.com/in/deviprasadchebodula) · [GitHub](https://github.com/deviprasadchebodula)
