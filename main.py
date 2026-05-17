from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pickle
import pandas as pd
import shap
import numpy as np
import csv
import io

from database import engine, get_db, Base
import models
from auth import hash_password, verify_password, create_token, decode_token

Base.metadata.create_all(bind=engine)

app = FastAPI()
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

model = pickle.load(open("model.pkl", "rb"))
explainer = shap.TreeExplainer(model)

FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences"
]


# ---------- Auth schemas ----------

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str


# ---------- Auth dependency ----------

def get_current_faculty(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return decode_token(credentials.credentials)


# ---------- Auth endpoints ----------

@app.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(models.Faculty).filter(models.Faculty.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    db.add(models.Faculty(
        name=req.name,
        email=req.email,
        hashed_password=hash_password(req.password)
    ))
    db.commit()
    return {"message": "Account created successfully"}

@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    faculty = db.query(models.Faculty).filter(models.Faculty.email == req.email).first()
    if not faculty or not verify_password(req.password, faculty.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token({"sub": faculty.email, "name": faculty.name})
    return {"access_token": token, "name": faculty.name}

@app.get("/me")
def get_me(current: dict = Depends(get_current_faculty)):
    return {"email": current["sub"], "name": current["name"]}

@app.get("/dashboard")
def get_dashboard(current: dict = Depends(get_current_faculty), db: Session = Depends(get_db)):
    records = db.query(models.Prediction)\
                .filter(models.Prediction.faculty_email == current["sub"])\
                .order_by(models.Prediction.created_at.desc())\
                .all()
    return [
        {
            "time": r.created_at.strftime("%d %b, %H:%M"),
            "age": r.age,
            "failures": r.failures,
            "absences": r.absences,
            "studytime": r.studytime,
            "at_risk": r.at_risk,
            "probability": r.probability
        }
        for r in records
    ]


# ---------- Student schema ----------

class Student(BaseModel):
    school: int
    sex: int
    age: int
    address: int
    famsize: int
    Pstatus: int
    Medu: int
    Fedu: int
    Mjob: int
    Fjob: int
    reason: int
    guardian: int
    traveltime: int
    studytime: int
    failures: int
    schoolsup: int
    famsup: int
    paid: int
    activities: int
    nursery: int
    higher: int
    internet: int
    romantic: int
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int


# ---------- Intervention logic ----------

def generate_interventions(s: dict, shap: dict) -> list:
    result = []
    if s["failures"] >= 2:
        result.append({"type": "Academic", "priority": "High", "action": "Enroll in remedial classes immediately and assign a subject mentor"})
    elif s["failures"] == 1:
        result.append({"type": "Academic", "priority": "Medium", "action": "Schedule weekly one-on-one tutoring sessions"})
    if s["absences"] > 10:
        result.append({"type": "Attendance", "priority": "High", "action": "Contact parents and initiate a formal attendance improvement plan"})
    elif s["absences"] > 5:
        result.append({"type": "Attendance", "priority": "Medium", "action": "Issue attendance warning and monitor weekly"})
    if s["studytime"] <= 1:
        result.append({"type": "Study Plan", "priority": "High", "action": "Create a structured daily study schedule with faculty guidance"})
    if s["Dalc"] >= 3 or s["Walc"] >= 4:
        result.append({"type": "Counselling", "priority": "High", "action": "Refer to student counsellor for substance use discussion"})
    if s["health"] <= 2:
        result.append({"type": "Health", "priority": "Medium", "action": "Refer to student health services for a checkup"})
    if s["famrel"] <= 2:
        result.append({"type": "Family", "priority": "Medium", "action": "Schedule parent-teacher meeting and family counselling session"})
    if s["higher"] == 0:
        result.append({"type": "Motivation", "priority": "Medium", "action": "Career counselling session to help student set academic goals"})
    if s["internet"] == 0:
        result.append({"type": "Resources", "priority": "Low", "action": "Provide access to school computer lab and library resources"})
    if s["goout"] >= 4 and s["freetime"] >= 4:
        result.append({"type": "Time Management", "priority": "Medium", "action": "Enroll in time management workshop and create structured activity plan"})
    if not result:
        result.append({"type": "Monitoring", "priority": "Low", "action": "No immediate intervention needed — continue regular monitoring"})
    return result


# ---------- Predict endpoint (protected) ----------

@app.post("/predict")
def predict(student: Student, current: dict = Depends(get_current_faculty), db: Session = Depends(get_db)):
    data = pd.DataFrame([student.model_dump()], columns=FEATURES)
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    shap_values = explainer.shap_values(data)
    shap_dict = dict(zip(FEATURES, np.round(shap_values[0], 3).tolist()))
    interventions = generate_interventions(student.model_dump(), shap_dict)

    db.add(models.Prediction(
        faculty_email=current["sub"],
        age=student.age,
        absences=student.absences,
        failures=student.failures,
        studytime=student.studytime,
        at_risk=prediction,
        probability=round(probability * 100, 1)
    ))
    db.commit()

    return {
        "at_risk": prediction,
        "risk_probability": round(probability * 100, 1),
        "shap_explanation": shap_dict,
        "interventions": interventions
    }


# ---------- CSV Template Download ----------

@app.get("/template-csv")
def download_template():
    content = (
        "name,age,sex,studytime,failures,absences,higher,schoolsup,"
        "internet,romantic,health,famrel,freetime,goout,Dalc,Walc,Medu,Fedu,famsup\n"
        "John Doe,17,M,2,0,4,yes,no,yes,no,3,4,3,3,1,1,3,2,yes\n"
        "Jane Smith,16,F,1,1,12,yes,yes,no,yes,2,2,4,4,2,3,2,1,no\n"
        "Ravi Kumar,18,M,3,0,2,yes,no,yes,no,4,5,2,2,1,1,4,3,yes\n"
    )
    return StreamingResponse(
        io.StringIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=student_template.csv"}
    )


# ---------- CSV Batch Predict ----------

def encode_row(row: dict) -> dict:
    def yn(v): return 1 if str(v).strip().lower() == 'yes' else 0
    def sex_enc(v): return 1 if str(v).strip().upper() == 'M' else 0
    return {
        "school": 0, "address": 1, "famsize": 0, "Pstatus": 1,
        "Mjob": 2, "Fjob": 2, "reason": 0, "guardian": 1,
        "traveltime": 1, "paid": 0, "activities": 0, "nursery": 1,
        "sex":       sex_enc(row.get("sex", "F")),
        "age":       int(row.get("age", 17)),
        "Medu":      int(row.get("Medu", 2)),
        "Fedu":      int(row.get("Fedu", 2)),
        "studytime": int(row.get("studytime", 2)),
        "failures":  int(row.get("failures", 0)),
        "schoolsup": yn(row.get("schoolsup", "no")),
        "famsup":    yn(row.get("famsup", "yes")),
        "higher":    yn(row.get("higher", "yes")),
        "internet":  yn(row.get("internet", "yes")),
        "romantic":  yn(row.get("romantic", "no")),
        "famrel":    int(row.get("famrel", 4)),
        "freetime":  int(row.get("freetime", 3)),
        "goout":     int(row.get("goout", 3)),
        "Dalc":      int(row.get("Dalc", 1)),
        "Walc":      int(row.get("Walc", 1)),
        "health":    int(row.get("health", 3)),
        "absences":  int(row.get("absences", 0)),
    }


@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    current: dict = Depends(get_current_faculty),
    db: Session = Depends(get_db)
):
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    results = []

    for row in reader:
        try:
            student_data = encode_row(row)
            data = pd.DataFrame([student_data], columns=FEATURES)
            prediction = int(model.predict(data)[0])
            probability = float(model.predict_proba(data)[0][1])
            shap_vals = explainer.shap_values(data)
            shap_dict = dict(zip(FEATURES, np.round(shap_vals[0], 3).tolist()))
            interventions = generate_interventions(student_data, shap_dict)

            db.add(models.Prediction(
                faculty_email=current["sub"],
                age=student_data["age"],
                absences=student_data["absences"],
                failures=student_data["failures"],
                studytime=student_data["studytime"],
                at_risk=prediction,
                probability=round(probability * 100, 1)
            ))

            results.append({
                "name": row.get("name", "Student"),
                "age": student_data["age"],
                "failures": student_data["failures"],
                "absences": student_data["absences"],
                "at_risk": prediction,
                "risk_probability": round(probability * 100, 1),
                "top_intervention": interventions[0]["action"],
                "interventions": interventions
            })
        except Exception as e:
            results.append({"name": row.get("name", "?"), "error": str(e)})

    db.commit()
    return results
