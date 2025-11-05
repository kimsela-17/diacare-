
import streamlit as st
import joblib
import pandas as pd
import random

# Load model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_classifier_pipeline.joblib")

model = load_model()

st.set_page_config(page_title="Diacare - Diabetes Prediction Assistant", page_icon="💉", layout="centered")

st.title("💉 Diacare: Diabetes Prediction & Health Assistant")
st.write("Predicts if a person is Normal, Prediabetic, or Diabetic and provides personalized advice.")

# Sidebar inputs
st.sidebar.header("🧍‍♀️ Enter Your Health Information")
glucose = st.sidebar.number_input("Glucose Level (mg/dL)", 0, 300, 100)
bmi = st.sidebar.number_input("Body Mass Index (BMI)", 0.0, 70.0, 25.0)
age = st.sidebar.number_input("Age", 1, 120, 30)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", 0, 200, 80)
insulin = st.sidebar.number_input("Insulin (mu U/mL)", 0, 900, 85)
skin_thickness = st.sidebar.number_input("Skin Thickness (mm)", 0, 100, 20)
pregnancies = st.sidebar.number_input("Pregnancies (if applicable)", 0, 20, 1)

input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'Age': [age]
})

# Prediction
st.header("🔍 Diabetes Prediction")
if st.button("Predict Diabetes Level"):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    labels = ["Normal", "Prediabetes", "Diabetes"]
    predicted_label = labels[int(prediction)]
    
    st.subheader(f"🧠 Prediction: {predicted_label}")
    st.progress(int(probs[int(prediction)] * 100))
    st.write(f"Confidence: **{probs[int(prediction)]*100:.1f}%**")

    # Recommendations
    if predicted_label == "Normal":
        st.success("✅ You're healthy! Maintain regular checkups and a balanced diet.")
        st.write("""**Medical Advice:** Continue with annual medical checkups.
**Lifestyle Tips:** Stay active, eat balanced meals, and get enough sleep.
**Treatment:** None needed.
**Message:** Keep up the great work!""")
        
    elif predicted_label == "Prediabetes":
        st.warning("⚠️ You may be at risk of diabetes.")
        st.write("""**Medical Advice:** Consult a doctor early.
**Lifestyle Tips:** Exercise daily, reduce sugar intake.
**Treatment:** Lifestyle modification may help.
**Message:** Early action can prevent diabetes progression.""")

    elif predicted_label == "Diabetes":
        st.error("🚨 Diabetes detected.")
        st.write("""**Medical Advice:** See a healthcare provider.
**Lifestyle Tips:** Manage diet, follow medications.
**Treatment:** Insulin or oral medication as prescribed.
**Message:** Early management is key.""")

# Chatbot
st.markdown("---")
st.header("💬 Chat with Diacare")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask Diacare something about diabetes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = ""
    if "diet" in prompt.lower():
        response = "🍎 Eat vegetables, lean protein, avoid sugary foods."
    elif "exercise" in prompt.lower():
        response = "🏃 Regular exercise helps control blood sugar."
    elif "treatment" in prompt.lower():
        response = "💊 Treatment depends on diabetes type."
    elif "prediabetes" in prompt.lower():
        response = "⚠️ Prediabetes can be reversed with lifestyle changes."
    elif "normal" in prompt.lower():
        response = "✅ You are in a healthy range."
    elif "hello" in prompt.lower() or "hi" in prompt.lower():
        response = "👋 Hello! I am Diacare, your diabetes assistant."
    else:
        response = random.choice([
            "I’m not sure, please consult a doctor.",
            "Maintain a healthy lifestyle.",
            "Could you clarify your question?"
        ])
        
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

st.markdown("---")
st.caption("Developed for Hackathon 🧑‍💻 | Powered by Streamlit & AI")
