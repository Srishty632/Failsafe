import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("data/student-mat.csv", sep=";")
df["at_risk"] = (df["G3"] < 10).astype(int)

le = LabelEncoder()
text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["G3", "G2", "G1", "at_risk"])

model = pickle.load(open("model.pkl", "rb"))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP explanations generated!")
print("Plotting top features that affect student risk...")

shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
print("Chart saved as shap_summary.png!")
