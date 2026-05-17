import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/student-mat.csv", sep=";")

# If final grade < 10, student is at risk (1 = at risk, 0 = safe)
df["at_risk"] = (df["G3"] < 10).astype(int)

print("At risk students:", df["at_risk"].sum())
print("Safe students:", (df["at_risk"] == 0).sum())

# Convert text columns (like "yes"/"no") into numbers
le = LabelEncoder()
text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = le.fit_transform(df[col])

# Separate features (inputs) and label (output)
X = df.drop(columns=["G3", "G2", "G1", "at_risk"])
y = df["at_risk"]

# Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nData is ready!")
