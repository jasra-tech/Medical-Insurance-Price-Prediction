# importing Necessary Libraries
import numpy as np
import pickle as pkl
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Load trained model
model = pkl.load(open('MIPML.pkl', 'rb'))

st.set_page_config(page_title="Medical Insurance Predictor", layout="wide")

# --- Logo & Header ---
st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="100" alt="Logo">
        <h1 style="margin-top:10px; font-family:Arial, sans-serif;">Medical Insurance Price Predictor</h1>
        <p style="font-size:16px; color:gray; font-family:Arial, sans-serif;">
            Enter client details to calculate estimated premiums, analyze risk, and download a detailed report.
        </p>
        <hr style="border: 1px solid #e0e0e0;">
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Welcome Message & App Description ---
st.markdown(
    """
    <div style="padding:20px; font-family:Arial, sans-serif;">
        <h2 style="color:#2c3e50;">👋 Welcome!</h2>
        <p style="font-size:15px; color:#34495e;">
            This application helps insurance agents and clients easily estimate medical insurance premiums 
            based on personal details such as age, gender, BMI, smoking status, and region.
        </p>
        <p style="font-size:15px; color:#34495e;">
            With just a few inputs, the system uses a trained machine learning model to predict premium costs, 
            visualize the breakdown, and generate a detailed PDF report for record-keeping or client sharing.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar for inputs ---
st.sidebar.header("Client Information")

client_name = st.sidebar.text_input("Client Name", "Client")
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
smoker = st.sidebar.selectbox('Smoker Status', ['Yes', 'No'])
region = st.sidebar.selectbox('Region', ['Southeast', 'Southwest', 'Northeast', 'Northwest'])
age = st.sidebar.slider('Age', 5, 80, 25)
bmi = st.sidebar.slider('BMI', 5, 100, 25)
children = st.sidebar.slider('Number of Children', 0, 5, 0)

# Encode categorical values
gender_val = 0 if gender == 'Female' else 1
smoker_val = 0 if smoker == 'No' else 1
region_mapping = {'Southeast': 0, 'Southwest': 1, 'Northeast': 2, 'Northwest': 3}
region_val = region_mapping[region]

# Prepare numeric input
input_data = np.asarray((age, gender_val, bmi, children, smoker_val, region_val)).reshape(1, -1)

# Prediction
if st.sidebar.button('Predict'):
    predicted_prem = model.predict(input_data)

    # USD and INR conversion
    usd_premium = round(predicted_prem[0], 2)
    inr_premium = round(usd_premium * 83, 2)

    st.subheader("Predicted Insurance Premium")
    st.write(f"**Annual:** ${usd_premium} (~₹{inr_premium})")
    st.write(f"**Half-Yearly:** ${round(usd_premium/2,2)} (~₹{round(inr_premium/2,2)})")
    st.write(f"**Quarterly:** ${round(usd_premium/4,2)} (~₹{round(inr_premium/4,2)})")
    st.write(f"**Monthly:** ${round(usd_premium/12,2)} (~₹{round(inr_premium/12,2)})")

    # --- Graph visualization (Premium Breakdown) ---
    labels = ['Annual', 'Half-Yearly', 'Quarterly', 'Monthly']
    values = [usd_premium, usd_premium/2, usd_premium/4, usd_premium/12]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Premium Breakdown (USD)")
    ax.set_ylabel("Premium ($)")
    st.pyplot(fig)

    # --- Smoker Risk Factor Graph ---
    smoker_input = np.asarray((age, gender_val, bmi, children, 1, region_val)).reshape(1, -1)
    nonsmoker_input = np.asarray((age, gender_val, bmi, children, 0, region_val)).reshape(1, -1)

    smoker_pred = model.predict(smoker_input)[0]
    nonsmoker_pred = model.predict(nonsmoker_input)[0]

    fig2, ax2 = plt.subplots()
    ax2.bar(['Non-Smoker', 'Smoker'], [nonsmoker_pred, smoker_pred], color=['green', 'red'])
    ax2.set_title("Smoker vs Non-Smoker Annual Premium")
    ax2.set_ylabel("Premium ($)")
    st.pyplot(fig2)

    # --- PDF Report Generation using BytesIO (Windows-safe) ---
    def create_pdf_bytes():
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(200, 750, "Insurance Premium Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 700, f"Client Name: {client_name}")
        c.drawString(50, 680, f"Age: {age}")
        c.drawString(50, 660, f"Gender: {gender}")
        c.drawString(50, 640, f"Smoker: {smoker}")
        c.drawString(50, 620, f"Region: {region}")
        c.drawString(50, 600, f"BMI: {bmi}")
        c.drawString(50, 580, f"Children: {children}")

        c.drawString(50, 540, f"Predicted Annual Premium: ${usd_premium} (~₹{inr_premium})")
        c.drawString(50, 520, f"Half-Yearly: ${round(usd_premium/2,2)} (~₹{round(inr_premium/2,2)})")
        c.drawString(50, 500, f"Quarterly: ${round(usd_premium/4,2)} (~₹{round(inr_premium/4,2)})")
        c.drawString(50, 480, f"Monthly: ${round(usd_premium/12,2)} (~₹{round(inr_premium/12,2)})")

        c.drawString(50, 440, "Smoker Risk Factor Comparison:")
        c.drawString(70, 420, f"Non-Smoker: ${round(nonsmoker_pred,2)}")
        c.drawString(70, 400, f"Smoker: ${round(smoker_pred,2)}")

        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer

    pdf_bytes = create_pdf_bytes()
    st.download_button(
        label="⬇️ Download Report as PDF",
        data=pdf_bytes,
        file_name=f"{client_name}_insurance_report.pdf",
        mime="application/pdf"
    )
