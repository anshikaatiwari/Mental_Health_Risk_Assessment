
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

def convert_to_float32(X):
    return X.astype('float32')

# Load trained model pipeline
model = joblib.load("depression_pipeline.pkl")

print(model.feature_names_in_)

# Match the exact feature names your model was trained on
feature_order = [
    'Gender', 'Age', 'City', 'Working Professional or Student',
       'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA',
       'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration',
       'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
       'Work/Study Hours', 'Financial Stress',
       'Family History of Mental Illness', 'Age_WorkPressure'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        age = int(form.get('age', 0))
        work_pressure = int(form.get('work_pressure', 0))

        # Match input to trained feature names
        input_data = pd.DataFrame([{
           
            'Age': age,
            'Gender': form.get('gender', ''),
            'City': form.get('city', ''),
            'Working Professional or Student': form.get('profession1', ''),
            'Degree': form.get('degree', ''),
            'Profession': form.get('profession2', ''),  # ADD THIS
            'Sleep Duration': form.get('sleep_duration', ''),
            'Dietary Habits': form.get('diet', ''),
            'CGPA': float(form.get('cgpa', 0.0)),
            'Work/Study Hours': int(form.get('study_hours', 0)),
            'Job Satisfaction': int(form.get('job_satisfaction', 0)),
            'Academic Pressure': int(form.get('academic_pressure', 0)),
            'Work Pressure': work_pressure,
            'Financial Stress': int(form.get('financial_stress', 0)),
            'Have you ever had suicidal thoughts ?': form.get('suicidal_thoughts', ''),
            'Family History of Mental Illness': form.get('family_mental_illness', ''),
            'Study Satisfaction': int(form.get('study_satisfaction', 0)),
            'Age_WorkPressure': age * work_pressure
        }])

        input_data = input_data[feature_order]  # Ensure column order matches training

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if probability > 0.7:
         result_text = "High risk detected"
         advice = (
            "⚠️ Please consider talking to a mental health professional.")
        elif 0.5 <= probability <= 0.7:
            result_text = "Moderate risk detected "
            advice = (
            "Practice self care activities like meditation and journaling 🧘🏻‍♂📝."
            "Consider taking a professional help!")
        else:
            result_text = "Low risk detected."
            advice = (
            "You Are completely fine 🥳")
            
        note=("Note: This assessment is not a diagnostic tool."
            "The model has been evaluated for fairness across demographic groups, ages and gender. "
            "The individual results may vary.")


        return render_template(
            'result.html',
            prediction=result_text,
            probability=f"{probability:.2%}",
            advice=advice,
            note=note
        )
    
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

