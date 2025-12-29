import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt
import numpy as np

# تحميل البيانات
df = pd.read_csv("data/training_data_30.7.2025 (done) - نسخة.csv", sep=";")

# ===== 1) تصحيح القيم السالبة =====
cols_to_fix = ["salary", "monthly_expense", "debt_amount", "percentage", "job_years", "age"]
df[cols_to_fix] = df[cols_to_fix].abs()

# ===== 2) تحويل القيم النوعية إلى أرقام =====
df["late_payment"] = df["late_payment"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["has_workers"] = df["has_workers"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["has_dependents"] = df["has_dependents"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["secondary_salary"] = df["secondary_salary"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["housing_status"] = df["housing_status"].str.strip().str.lower().map({"own": 1, "rent": 0})
df["job_score"] = df["job_score"].str.strip().str.upper().map({"A": 3, "B": 2, "C": 1, "D": 0})
df["job_type"] = df["job_type"].str.strip().str.upper().map({"GOV": 1, "PRV": 0})
df["job_status"] = df["job_status"].str.strip().str.lower().map({"employeed": 1, "retired": 0})

# ===== 3) إنشاء المؤشرات المشتقة =====
df["late_payment_ratio"] = df["late_payment"] / (df["amount_loans"] + 1)
df["expense_ratio"] = df["monthly_expense"] / (df["salary"] + 1)
df["debt_to_income"] = df["debt_amount"] / (df["salary"] + 1)
df["financial_burden"] = (df["monthly_expense"] + df["debt_amount"]/12) / (df["salary"] + 1)
#df["net_balance"] = df["savings_balance"] - df["debt_amount"]
#df["loans_to_age"] = df["amount_loans"] / (df["age"] + 1)
#df["workers_but_low_income"] = ((df["has_workers"] == 1) & (df["salary"] < df["salary"].median())).astype(int)

# ===== 4) اختيار الميزات =====
features = [
    "late_payment_ratio",
    "expense_ratio",
    "debt_to_income",
   # "workers_but_low_income",

    "late_payment",
    "housing_status",
    "has_dependents",
    "amount_loans",
    #"loans_to_age",
   "financial_burden",
    "has_workers",
    "secondary_salary",
    "percentage",
    "job_type",
    "job_status",
    "job_years",
    "job_score",
]

X = df[features]
y = df["financial_distress"]

# ===== 5) تقسيم البيانات =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 6) موازنة البيانات وتطبيعها =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_scaled, y_train)

# ===== 7) تدريب نموذج XGBoost =====
model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=3,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_balanced, y_train_balanced)

# ===== 8) التقييم =====
y_pred = model.predict(X_test_scaled)
print("الدقة (Accuracy):", accuracy_score(y_test, y_pred))
print("تقرير التصنيف:\n", classification_report(y_test, y_pred))
print("مصفوفة الالتباس:\n", confusion_matrix(y_test, y_pred))

# ===== 9) ربط التنبؤ باسم المستخدم =====
df_test = X_test.copy()
df_test["username"] = df.loc[X_test.index, "username"]
df_test["financial_distress_prediction"] = y_pred
df_test["financial_distress_prediction"] = df_test["financial_distress_prediction"].map({0: "مستقر ماليًا", 1: "متعثر ماليًا"})

print(df_test[["username", "financial_distress_prediction"]].head())

# ===== 10) حفظ النموذج والمقياس =====
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ تم حفظ النموذج في models/model.pkl")

# ===== 11) عرض أهمية الميزات =====
importances = model.feature_importances_
features_array = np.array(features)
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Main Causes of Financial Distress")
plt.bar(range(len(features)), importances[indices], color="green")
plt.xticks(range(len(features)), features_array[indices], rotation=45)
plt.tight_layout()
plt.show()