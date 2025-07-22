import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle

# تحميل البيانات
df = pd.read_csv("data/financial_data.csv", sep=";")

# تحويل النصوص لأرقام
df["late_payment"] = df["late_payment"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["has_workers"] = df["has_workers"].str.strip().str.lower().map({"yes": 1, "no": 0})
df["housing_status"] = df["housing_status"].str.strip().str.lower().map({"rent": 0, "own": 1})
df["has_dependents"] = df["has_dependents"].str.strip().str.lower().map({"yes": 1, "no": 0})

# تصحيح القيم السالبة
cols_to_fix = ["salary", "monthly_expense", "debt_amount", "savings_balance"]
df[cols_to_fix] = df[cols_to_fix].abs()

# حساب المؤشرات
df["expense_ratio"] = df["monthly_expense"] / (df["salary"] + 1)
df["debt_to_income"] = df["debt_amount"] / (df["salary"] + 1)
df["net_balance"] = df["savings_balance"] - df["debt_amount"]

# اختيار الميزات
features = [
    "expense_ratio",
    "debt_to_income",
    "net_balance",
    "late_payment",
    "housing_status",
    "has_workers",
    "has_dependents",
    "num_loans"
]

X = df[features]
y = df["financial_distress"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# موازنة البيانات باستخدام SMOTE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train_scaled, y_train)

# تدريب الموديل
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# التقييم
y_pred = model.predict(X_test_scaled)
print("🎯 الدقة:", accuracy_score(y_test, y_pred))
print("📋 تقرير النموذج:\n", classification_report(y_test, y_pred))

# ربط التنبؤ مع اسم المستخدم
df_test = X_test.copy()
df_test["username"] = df.loc[X_test.index, "username"]
df_test["financial_distress_prediction"] = y_pred
df_test["financial_distress_prediction"] = df_test["financial_distress_prediction"].map({0: "✅ سليم", 1: "⚠ متعثر"})

print(df_test[["username", "financial_distress_prediction"]].head())

# حفظ الموديل والـ scaler
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ تم حفظ النموذج في models/model.pkl")