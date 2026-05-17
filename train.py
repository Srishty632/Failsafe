import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle

df = pd.read_csv("data/student-mat.csv", sep=";")
df["at_risk"] = (df["G3"] < 10).astype(int)

le = LabelEncoder()
text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["G3", "G2", "G1", "at_risk"])
y = df["at_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "At Risk"]))

# Save the model so we can use it later
pickle.dump(model, open("model.pkl", "wb"))
print("\nModel saved as model.pkl!")
