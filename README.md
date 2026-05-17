# FAILSAFE — Student Risk Prediction System

> **Predicting student failure before it happens — with Explainable AI and personalised interventions.**

FAILSAFE is a web-based early warning system for educational institutions. Faculty can upload student data and receive ML-powered failure risk predictions backed by SHAP explanations, with auto-generated personalised intervention plans — enabling timely action before it's too late.

---

## Features

- **Faculty Authentication** — Secure register/login system with JWT-based session management
- **Single Student Prediction** — Enter student details via a form and get an instant risk score with a visual gauge
- **Batch CSV Upload** — Upload an entire class at once and get risk predictions for all students in one go
- **SHAP Explainability** — Every prediction is backed by SHAP values showing exactly which factors drove the result, making it trustworthy and actionable for non-technical faculty
- **Personalised Intervention Plans** — Auto-generated, prioritised intervention recommendations per student (academic support, counselling, attendance plans, etc.)
- **Faculty Dashboard** — View prediction history, total students analysed, at-risk count, and average risk percentage
- **CSV Template Download** — Downloadable template so faculty can fill and upload data in the correct format
- **Printable Reports** — Each prediction result can be printed or saved as a PDF report

---

## Tech Stack

**Machine Learning**
- Python, XGBoost, scikit-learn, SHAP, Pandas, Matplotlib

**Backend**
- FastAPI, SQLAlchemy, SQLite, JWT Authentication (python-jose), bcrypt

**Frontend**
- HTML, CSS, JavaScript (Vanilla)

**Dataset**
- UCI Student Performance Dataset (`student-mat.csv`) — available on [Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)

---

## Project Structure

```
failsafe/
├── main.py               # FastAPI backend — all API endpoints
├── auth.py               # JWT token creation and password hashing
├── database.py           # SQLAlchemy database setup (SQLite)
├── models.py             # Database models (Faculty, Prediction)
├── train_model.py        # XGBoost model training script
├── shap_explain.py       # SHAP summary plot generation
├── model.pkl             # Trained XGBoost model
├── shap_summary.png      # SHAP feature importance chart
├── failsafe.db           # SQLite database (auto-created)
├── requirements.txt      # Python dependencies
├── data/
│   └── student-mat.csv   # Dataset
└── frontend/
    ├── index.html        # Main app (Predict, Batch Upload, Dashboard)
    └── login.html        # Login and Register page
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Srishty632/Failsafe.git
cd Failsafe
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download `student-mat.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption) and place it inside a `data/` folder:

```
failsafe/data/student-mat.csv
```

### 5. Train the Model

```bash
python train_model.py
```

This generates `model.pkl` and prints accuracy and classification report.

### 6. Run the Backend Server

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`

### 7. Open the Frontend

Open `frontend/login.html` in your browser, register an account, and start using FAILSAFE.

---

## Model Details

- **Algorithm:** XGBoost Classifier
- **Target:** `at_risk` — 1 if final grade (G3) < 10, else 0
- **Features:** 30 student attributes including attendance, study time, family background, lifestyle, and past failures
- **Train/Test Split:** 80% / 20%
- **Explainability:** SHAP TreeExplainer — top 8 contributing features shown per prediction

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Register a new faculty account |
| POST | `/login` | Login and receive JWT token |
| GET | `/me` | Get current logged-in faculty info |
| POST | `/predict` | Predict risk for a single student |
| POST | `/predict-csv` | Batch predict from uploaded CSV file |
| GET | `/dashboard` | Get prediction history for logged-in faculty |
| GET | `/template-csv` | Download sample CSV template |

---

## How It Works

1. Faculty registers and logs in securely
2. Enter a student's academic, personal, lifestyle, and family details — or upload a CSV for the whole class
3. XGBoost model predicts failure risk and outputs a probability score
4. SHAP values explain which features contributed most to the prediction
5. The system auto-generates prioritised interventions (e.g. remedial classes, counselling referral, attendance plan)
6. All predictions are saved to the faculty's dashboard for tracking

---

## Screenshots

> *(Add screenshots of the login page, prediction form, result panel with SHAP bars, and dashboard here)*

---

## Mentor

**Avni Jhalani** — 6378972643

---

## License

This project was built for academic submission purposes.