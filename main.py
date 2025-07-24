import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import time
#hay jana

st.set_page_config(page_title="نموذج التعثر المالي", page_icon="📉", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f5f9ff;
    }
    .main {
        background-color: #e6f0ff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 10px #b3cfff;
    }
    .css-1d391kg {background-color: #e6f0ff;}
    </style>
""", unsafe_allow_html=True)

st.title("🏦 نظام التنبؤ بالتعثر المالي")
st.markdown("### 🧾 ارفع ملف بيانات العملاء لتحليل حالتهم المالية 🔍")

uploaded_file = st.file_uploader("📂 ارفع ملف CSV بصيغة ; (فاصلة منقوطة)", type=["csv"])

def animated_counter(container, target, color):
    # عداد بسيط يحرك الرقم من 0 إلى target
    for i in range(target + 1):
        container.markdown(f"<h2 style='color:{color}; font-weight:bold;'>{i}</h2>", unsafe_allow_html=True)
        time.sleep(0.02)

if uploaded_file:
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        df = pd.read_csv(uploaded_file, sep=";")

        df["late_payment"] = df["late_payment"].str.strip().str.lower().map({"yes": 1, "no": 0})
        df["has_workers"] = df["has_workers"].str.strip().str.lower().map({"yes": 1, "no": 0})
        df["housing_status"] = df["housing_status"].str.strip().str.lower().map({"rent": 0, "own": 1})
        df["has_dependents"] = df["has_dependents"].str.strip().str.lower().map({"yes": 1, "no": 0})

        df["expense_ratio"] = df["monthly_expense"] / (df["salary"] + 1)
        df["debt_to_income"] = df["debt_amount"] / (df["salary"] + 1)
        df["net_balance"] = df["savings_balance"] - df["debt_amount"]

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

        X_new = df[features]
        X_scaled = scaler.transform(X_new)

        predictions = model.predict(X_scaled)
        df["financial_distress_prediction"] = predictions
        df["financial_distress_prediction"] = df["financial_distress_prediction"].map({0: "✅ سليم", 1: "⚠ متعثر"})

        # نحسب الأعداد
        total_salim = (df["financial_distress_prediction"] == "✅ سليم").sum()
        total_mutaathir = (df["financial_distress_prediction"] == "⚠ متعثر").sum()

        # نعرضهم فوق الصفحة بألوان وحركة
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='color:green;'>✅ السليمون</h3>", unsafe_allow_html=True)
            counter_container_salim = st.empty()
            animated_counter(counter_container_salim, total_salim, "green")
        with col2:
            st.markdown("<h3 style='color:red;'>⚠ المتعثرون</h3>", unsafe_allow_html=True)
            counter_container_mutaathir = st.empty()
            animated_counter(counter_container_mutaathir, total_mutaathir, "red")

        st.success("✅ تم التحليل بنجاح!")
        # عرض فقط username والنتيجة المطلوبة
        st.dataframe(df[["username", "financial_distress_prediction"]], use_container_width=True)

        csv_output = df[["username", "financial_distress_prediction"]].to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 تحميل النتائج",
            data=csv_output,
            file_name="financial_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ حصل خطأ: {e}")
else:
    st.info("👆 الرجاء رفع ملف بصيغة CSV")

st.markdown("""---""")
st.markdown("<center>صُمم بحب 💙 لأبطال طويق والإنماء</center>", unsafe_allow_html=True)