import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

st.set_page_config(page_title="كشف التعثر المالي", page_icon="📉", layout="wide")

st.markdown("""
<style>
@font-face {
   font-family: Nunito-Medium;
   src: url(/fonts/Nunito-Medium.ttf);
}
@font-face {
   font-family: Nunito-Black;
   src: url(/fonts/Nunito-Black.ttf);
} 
   .stApp {
   background-color: #FDF8F5;
   display: flex;
   align-items: center;
   justify-content: center;
   text-align: center;
   font-family: Nunito-Medium !important;
   margin: 0;
   padding: 0;
   box-sizing: border-box;
   }
   h1 {
   color: #002134 !important;
   font-family: Nunito-Black;
   }
   h3 {
    font-family: Nunito-Black;
    color: #837FD8 !important;
   }
   .CSV {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    color: #002134 !important;
    font-family: Nunito-Black !important;
    margin: 10px;
    direction: rtl;
    text-align: right;
    box-shadow: 0 3px 8px rgba(0,0,0,0.15);
   }
   footer {
    color: #002134 !important;
   }
</style>
""", unsafe_allow_html=True)

st.title("نظام التنبؤ بالتعثر المالي")
st.markdown("<h3>ارفع ملف بيانات العملاء لتحليل حالتهم المالية</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

def animated_counter(container, target, color):
   for i in range(target - 10, target + 1):
       container.markdown(f"<h1 style='color: {color}'>{i}</h1>", unsafe_allow_html=True)
       time.sleep(0.05)

if uploaded_file:
   try:
       with open("models/model.pkl", "rb") as f:
           model = pickle.load(f)

       with open("models/scaler.pkl", "rb") as f:
           scaler = pickle.load(f)

       df = pd.read_csv(uploaded_file, sep=";")

       df["late_payment"] = df["late_payment"].str.strip().str.lower().map({"yes": 1, "no": 0})
       df["has_workers"] = df["has_workers"].str.strip().str.lower().map({"yes": 1, "no": 0})
       df["has_dependents"] = df["has_dependents"].str.strip().str.lower().map({"yes": 1, "no": 0})
       df["secondary_salary"] = df["secondary_salary"].str.strip().str.lower().map({"yes": 1, "no": 0})
       df["housing_status"] = df["housing_status"].str.strip().str.lower().map({"own": 1, "rent": 0})
       df["job_score"] = df["job_score"].str.strip().str.upper().map({"A": 3, "B": 2, "C": 1, "D": 0})
       df["job_type"] = df["job_type"].str.strip().str.upper().map({"GOV": 1, "PRV": 0})
       df["job_status"] = df["job_status"].str.strip().str.lower().map({"employeed": 1, "retired": 0})

       df["late_payment_ratio"] = df["late_payment"] / (df["amount_loans"] + 1)
       df["expense_ratio"] = df["monthly_expense"] / (df["salary"] + 1)
       df["debt_to_income"] = df["debt_amount"] / (df["salary"] + 1)
       df["financial_burden"] = (df["monthly_expense"] + df["debt_amount"] / 12) / (df["salary"] + 1)

       features = [
           "late_payment_ratio", "expense_ratio", "debt_to_income", "late_payment",
           "housing_status", "has_dependents", "amount_loans", "financial_burden",
           "has_workers", "secondary_salary", "percentage", "job_type", "job_status", "job_score",
       ]
       X_new = df[features]
       X_scaled = scaler.transform(X_new)
       predictions = model.predict(X_scaled)

       df["financial_distress_prediction"] = predictions
       df["financial_distress_prediction"] = df["financial_distress_prediction"].map({0: "مستقر ماليًا", 1: "متعثر ماليًا"})

       def determine_reason(row):
           if row["financial_distress_prediction"] != "متعثر ماليًا":
               return "لا يوجد"

           reasons = []
           if row["expense_ratio"] > 0.7:
               reasons.append("معدل صرف عالي")
           if row["late_payment_ratio"] > 0.2:
               reasons.append("تأخير متكرر بالسداد")
           if row["debt_to_income"] > 0.6:
               reasons.append("نسبة الدين مرتفعة")
           if row["financial_burden"] > 0.8:
               reasons.append("عبء مالي كبير")

           return " + ".join(reasons) if reasons else "غير محدد بدقة"


       df["reason"] = df.apply(determine_reason, axis=1)

       total_stable = (df["financial_distress_prediction"] == "مستقر ماليًا").sum()
       total_distressed = (df["financial_distress_prediction"] == "متعثر ماليًا").sum()

       col1, col2 = st.columns(2)
       with col1:
           st.markdown("<h3>مستقر ماليًا</h3>", unsafe_allow_html=True)
           animated_counter(st.empty(), total_stable, "#837FD8")
       with col2:
           st.markdown("<h3>متعثر ماليًا</h3>", unsafe_allow_html=True)
           animated_counter(st.empty(), total_distressed, "#837FD8")

       st.success("تم تحليل البيانات بنجاح!")
       st.dataframe(df[["username", "financial_distress_prediction", "reason"]], use_container_width=True)

       reason_counts = df[df["reason"] != "None"]["reason"].str.split(" + ").explode().value_counts()


       csv_upload = df.to_csv(index=False).encode('utf-8-sig')
       st.download_button(
           label="📥 تحميل النتائج",
           data=csv_upload,
           file_name="financial_predictions.csv",
           mime="text/csv"
       )


   except Exception as e:
       st.error(f" حصل خطأ: {e}")

else:
   st.markdown(
       """
       <div class="CSV">
           <p style="margin: 30px 10px 0px 0px;"><strong>ملاحظات قبل رفع الملف:</strong></p>
           <ul style="margin: 30px;">
               <li>يجب رفع الملف بصيغة (<span style="color: #837FD8;"><strong>CSV</strong></span>).</li>
               <li>تأكد من أن البيانات لا تحتوي على قيم مفقودة.</li>
               <li>يجب أن يحتوي الملف على الأعمدة الصحيحة:</li>
               <ol style="padding-right: 40px;">
                <li>أسم المستخدم (<span style="color: #837FD8;">username</span>)</li>
                <li>نوع الوظيفة (<span style="color: #837FD8;">job_type</span>)</li>
                <li>الوضع الوظيفي (<span style="color: #837FD8;">job_status</span>)</li>
                <li>سنوات الخدمة (<span style="color: #837FD8;">job_years</span>)</li>
                <li>تصنيف جهة العمل (<span style="color: #837FD8;">job_score</span>)</li>
                <li>الراتب (<span style="color: #837FD8;">salary</span>)</li>
                <li>وجود راتب آخر (<span style="color: #837FD8;">secondary_salary</span>)</li>
                <li>الصرف الشهري (<span style="color: #837FD8;">monthly_expense</span>)</li>
                <li>تأخر الدفع (<span style="color: #837FD8;">late_payment</span>)</li>
                <li>قيمة الدين (<span style="color: #837FD8;">debt_amount</span>)</li>
                <li>عدد القروض (<span style="color: #837FD8;">num_loans</span>)</li>
                <li>المتبقي من النسبة المسموحة للاستقطاع من الراتب (<span style="color: #837FD8;">percentage</span>)</li>
                <li>السكن (<span style="color: #837FD8;">housing_status</span>)</li>
                <li>وجود عمال (<span style="color: #837FD8;">has_workers</span>)</li>
                <li>وجود معالين (<span style="color: #837FD8;">has_dependents</span>)</li>
               </ol>
           </ul>
       </div>
       """,
       unsafe_allow_html=True
   )

st.markdown("---")
st.markdown("<footer>صمم من قبل فريق نرج ( أبرار الدخيل - غادة الدخيل - جنا الغامدي ) . هاكاثون أمد</footer>", unsafe_allow_html=True)

