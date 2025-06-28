import streamlit as st
import cloudpickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        with open("diabetes_pipeline.pkl", "rb") as f:
            pipeline = cloudpickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = cloudpickle.load(f)
        return pipeline, feature_names
    except Exception as e:
        st.error(f" Failed to load model: {e}")
        return None, None

# Load model
pipeline, feature_names = load_model()

# App title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on various health indicators.
Please enter your health information below to get a prediction.
""")

if pipeline is not None and feature_names is not None:
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Health Information")
        
        # Input fields based on typical diabetes dataset features
        pregnancies = st.number_input(
            "Number of Pregnancies", 
            min_value=0, 
            max_value=20, 
            value=0,
            help="Number of times pregnant"
        )
        
        glucose = st.number_input(
            "Glucose Level", 
            min_value=0, 
            max_value=300, 
            value=120,
            help="Plasma glucose concentration (mg/dL)"
        )
        
        blood_pressure = st.number_input(
            "Blood Pressure", 
            min_value=0, 
            max_value=200, 
            value=80,
            help="Diastolic blood pressure (mm Hg)"
        )
        
        skin_thickness = st.number_input(
            "Skin Thickness", 
            min_value=0, 
            max_value=100, 
            value=20,
            help="Triceps skin fold thickness (mm)"
        )
    
    with col2:
        st.subheader("📊 Additional Metrics")
        
        insulin = st.number_input(
            "Insulin Level", 
            min_value=0, 
            max_value=1000, 
            value=80,
            help="2-Hour serum insulin (mu U/ml)"
        )
        
        bmi = st.number_input(
            "BMI", 
            min_value=0.0, 
            max_value=70.0, 
            value=25.0,
            step=0.1,
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function", 
            min_value=0.0, 
            max_value=3.0, 
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function (genetic influence)"
        )
        
        age = st.number_input(
            "Age", 
            min_value=18, 
            max_value=120, 
            value=30,
            help="Age in years"
        )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Adjust column names if they don't match your feature names
    # You might need to modify this based on your actual feature names
    if len(feature_names) == 8:
        input_data.columns = feature_names
    
    # Prediction section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
            try:
                # Make prediction
                prediction = pipeline.predict(input_data)[0]
                prediction_proba = pipeline.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                if prediction == 1:
                    st.error(f"⚠️ **High Risk of Diabetes**")
                    st.error(f"Probability: {prediction_proba[1]:.2%}")
                else:
                    st.success(f"✅ **Low Risk of Diabetes**")
                    st.success(f"Probability of No Diabetes: {prediction_proba[0]:.2%}")
                
                # Show probability distribution
                st.markdown("### 📊 Risk Breakdown")
                prob_df = pd.DataFrame({
                    'Risk Level': ['No Diabetes', 'Diabetes'],
                    'Probability': [prediction_proba[0], prediction_proba[1]]
                })
                
                st.bar_chart(prob_df.set_index('Risk Level'))
                
                # Health recommendations
                st.markdown("### 💡 Health Recommendations")
                if prediction == 1:
                    st.markdown("""
                    - 🏥 **Consult a healthcare professional immediately**
                    - 🥗 **Maintain a healthy diet with low sugar intake**
                    - 🏃‍♂️ **Engage in regular physical activity**
                    - 📊 **Monitor blood glucose levels regularly**
                    - 💊 **Follow any prescribed medication regimen**
                    """)
                else:
                    st.markdown("""
                    - 🍎 **Continue maintaining a balanced diet**
                    - 🏃‍♀️ **Keep up with regular exercise**
                    - 📅 **Schedule regular health check-ups**
                    - ⚖️ **Maintain a healthy weight**
                    - 🚭 **Avoid smoking and limit alcohol consumption**
                    """)
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that your input data matches the expected format.")
    
    # Display input summary
    st.markdown("---")
    st.markdown("### 📝 Input Summary")
    st.dataframe(input_data, use_container_width=True)
    
    # Model information
    with st.expander("ℹ️ About this Model"):
        st.markdown("""
        This diabetes prediction model was trained using a Decision Tree Classifier on health data.
        
        **Features used for prediction:**
        - Pregnancies: Number of pregnancies
        - Glucose: Plasma glucose concentration
        - Blood Pressure: Diastolic blood pressure
        - Skin Thickness: Triceps skin fold thickness
        - Insulin: 2-Hour serum insulin
        - BMI: Body mass index
        - Diabetes Pedigree Function: Genetic influence
        - Age: Age in years
        
        **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
        Always consult with a healthcare provider for medical decisions.
        """)

else:
    st.error("Unable to load the model. Please ensure the model files are available.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | For educational purposes only</p>
    </div>
    """, 
    unsafe_allow_html=True
)