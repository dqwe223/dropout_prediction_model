import joblib
import pandas as pd
import streamlit as st

# Load the model
model = joblib.load('dropout_prediction_model.pkl')

# Set up the Streamlit app
st.title("Online Course Dropout Prediction")

# Get user input for each feature
session_duration = st.number_input("Session Duration (minutes)", min_value=0, value=45)
user_satisfaction = st.number_input("User Satisfaction (1-5)", min_value=1, max_value=5, value=3)
quiz_scores = st.number_input("Quiz Scores (0-100)", min_value=0, max_value=100, value=70)

# Create a DataFrame for the input data
new_data = pd.DataFrame({
    'SessionDuration': [session_duration],
    'UserSatisfaction': [user_satisfaction],
    'QuizScores': [quiz_scores]
})

# Check if there are additional features in the model that the input data lacks
required_features = model.feature_names_in_
for feature in required_features:
    if feature not in new_data.columns:
        new_data[feature] = 0  # Use 0, or replace with appropriate values

# Predict when the button is clicked
if st.button("Predict"):
    prediction = model.predict(new_data)
    status = "likely to drop out" if prediction[0] == 1 else "not likely to drop out"
    st.write(f"The model predicts the student is {status} from the online course.")
