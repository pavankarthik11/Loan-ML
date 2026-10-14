
import streamlit as st
import pandas as pd
import joblib
import os

from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import hf_hub_download

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="AI Loan Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------
# Load Environment Variables
# --------------------------------

load_dotenv()


def get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key)


# --------------------------------
# Groq Client
# --------------------------------

groq_api_key = get_secret("GROQ_API_KEY")

client = None

if groq_api_key:
    client = Groq(api_key=groq_api_key)

# --------------------------------
# Load Model from Hugging Face
# --------------------------------

@st.cache_resource
def load_models():
    hf_token = get_secret("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN is missing. Add it to Streamlit Cloud Secrets."
        )

    model_path = hf_hub_download(
        repo_id="PavanKarthik/loan-risk-model",
        filename="random_forest.pkl",
        repo_type="model",
        token=hf_token
    )

    preprocessor_path = hf_hub_download(
        repo_id="PavanKarthik/loan-risk-model",
        filename="preprocessor.pkl",
        repo_type="model",
        token=hf_token
    )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor


try:
    model, preprocessor = load_models()

except Exception as e:
    st.error("Unable to load the model files from Hugging Face.")
    st.code(str(e))
    st.stop()

# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.title("📋 Project Information")

st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Problem Type:** Binary Classification")
st.sidebar.write("**Dataset Size:** 593,994 Records")
st.sidebar.write("**Target:** Loan Repayment Prediction")

# --------------------------------
# Tabs
# --------------------------------

tab1, tab2 = st.tabs([
    "🏦 Loan Prediction",
    "🤖 Loan Assistant"
])

# --------------------------------
# Loan Prediction
# --------------------------------

with tab1:
    st.title("🏦 AI Loan Risk Assessment System")

    st.markdown("""
    This application predicts whether a customer is likely to repay a loan
    using a trained **Random Forest Machine Learning model**.
    """)

    st.write("Enter the customer details below and click **Predict**.")

    # --------------------------------
    # User Inputs
    # --------------------------------

    col1, col2 = st.columns(2)

    with col1:
        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0
        )

        debt_to_income_ratio = st.number_input(
            "Debt to Income Ratio",
            min_value=0.0,
            format="%.3f"
        )

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married"]
        )

        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Self-employed", "Unemployed"]
        )

    with col2:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        education_level = st.selectbox(
            "Education Level",
            [
                "High School",
                "Bachelor's",
                "Master's",
                "PhD"
            ]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Debt consolidation",
                "Home improvement",
                "Business",
                "Medical",
                "Education",
                "Other"
            ]
        )

    grade_subgrade = st.selectbox(
        "Loan Grade",
        [
            "A1", "A2", "A3", "A4", "A5",
            "B1", "B2", "B3", "B4", "B5",
            "C1", "C2", "C3", "C4", "C5",
            "D1", "D2", "D3", "D4", "D5",
            "E1", "E2", "E3", "E4", "E5",
            "F1", "F2", "F3", "F4", "F5",
            "G1", "G2", "G3", "G4", "G5"
        ]
    )

    # --------------------------------
    # Prediction
    # --------------------------------

    if st.button("🔍 Predict"):

        input_data = pd.DataFrame({
            "annual_income": [annual_income],
            "debt_to_income_ratio": [debt_to_income_ratio],
            "credit_score": [credit_score],
            "loan_amount": [loan_amount],
            "interest_rate": [interest_rate],
            "gender": [gender],
            "marital_status": [marital_status],
            "education_level": [education_level],
            "employment_status": [employment_status],
            "loan_purpose": [loan_purpose],
            "grade_subgrade": [grade_subgrade]
        })

        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)

        confidence = probability[0][1] * 100

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.success("✅ Loan is likely to be Paid Back")
        else:
            st.error("❌ High Risk of Default")

        st.subheader("Confidence")
        st.progress(int(confidence))

        st.write(
            f"**Probability of Loan Paid Back:** "
            f"{probability[0][1] * 100:.2f}%"
        )

        st.write(
            f"**Probability of Loan Default:** "
            f"{probability[0][0] * 100:.2f}%"
        )

        # --------------------------------
        # Risk Level
        # --------------------------------

        st.subheader("📌 Risk Level")

        if confidence >= 85:
            st.success("🟢 Low Risk")
        elif confidence >= 60:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

        # --------------------------------
        # Recommendation
        # --------------------------------

        st.subheader("🏦 Recommendation")

        if confidence >= 85:
            st.success("""
            ### Approve Loan

            Reason:
            - Strong repayment probability
            - Low risk customer
            - Financial profile appears stable
            """)

        elif confidence >= 60:
            st.warning("""
            ### Review Application

            Reason:
            - Medium repayment probability
            - Additional document verification is recommended
            """)

        else:
            st.error("""
            ### Reject / Manual Review

            Reason:
            - High default probability
            - Customer needs further financial assessment
            """)

        # --------------------------------
        # AI Explanation
        # --------------------------------

        st.subheader("🤖 AI Explanation")

        if client is None:
            st.warning(
                "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets "
                "to enable AI explanations."
            )

        else:
            prompt = f"""
            You are an AI assistant helping a bank loan officer.

            Customer Details:

            Annual Income: {annual_income}
            Debt to Income Ratio: {debt_to_income_ratio}
            Credit Score: {credit_score}
            Loan Amount: {loan_amount}
            Interest Rate: {interest_rate}
            Gender: {gender}
            Marital Status: {marital_status}
            Education Level: {education_level}
            Employment Status: {employment_status}
            Loan Purpose: {loan_purpose}
            Loan Grade: {grade_subgrade}

            Prediction:
            {"Loan is likely to be Paid Back" if prediction[0] == 1 else "High Risk of Default"}

            Confidence:
            {confidence:.2f}%

            Instructions:
            - Do not assume any currency.
            - Do not invent any facts.
            - Use only the information provided.
            - Explain in simple English.
            - Keep the response under 120 words.

            Return your answer using this format:

            ### Summary

            ### Positive Factors

            ### Possible Concerns

            ### Recommendation
            """

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                st.write(response.choices[0].message.content)

            except Exception as e:
                st.warning(
                    "The prediction succeeded, but the AI explanation "
                    "could not be generated."
                )
                st.caption(str(e))

# --------------------------------
# Loan Assistant
# --------------------------------

with tab2:
    st.header("🤖 Loan Assistant")

    question = st.text_input("Ask any loan related question")

    if st.button("Ask Assistant"):

        if question.strip():

            with st.spinner("Searching knowledge base..."):
                try:
                    # Import the chatbot only when it is requested.
                    from rag.chatbot import ask_loan_assistant

                    answer = ask_loan_assistant(question)
                    st.write(answer)

                except Exception as e:
                    st.error(
                        "Unable to get an answer from the Loan Assistant."
                    )
                    st.caption(str(e))

        else:
            st.warning("Please enter a question.")
