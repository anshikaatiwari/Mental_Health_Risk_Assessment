
# Mental Health Risk Prediction App

##  Project Overview

This project aims to predict the risk of depression based on demographic, lifestyle, academic/work-related, and psychological indicators. The goal is to provide an accessible tool that helps identify early signs of mental distress, especially in students and working professionals.

The app is powered by a machine learning model trained on kaggle data and deployed as a lightweight web application using Flask.

---

##  Dataset

**Source**: Kaggle Mental health dataset (https://www.kaggle.com/competitions/playground-series-s4e11/overview)

**Features included:**

* Demographics: Age, Gender, City
* Academic/Work: CGPA, Degree, Profession, Study/Work hours
* Lifestyle: Sleep duration, Dietary habits
* Mental Health Indicators: Suicidal thoughts, Family history of mental illness, Satisfaction and Pressure levels

> **Preprocessing steps:**
>
> * Null/missing value handling (SimpleImputer)
> * Encoding categorical variables (OrdinalEncoder)
> * Feature scaling (StandardScaler)

---

## 🧠 Key Insights from Data

* **Sleep duration** and **academic/work pressure** had strong correlations with depression risk.
* Students with low study satisfaction and high CGPA often reported signs of stress.
* Suicidal Thoughts and financial stress were among the top indicators of elevated risk.
* Family mental health history was a significant factor in overall prediction probability.

---

##  Models Trained and Evaluated

* **Gradient Boosting Classifier**
* **Random Forest Classifier**
* **Logistic Regression**
* **K Nearest Neighbours**


Each model was trained using a pipeline built with `scikit-learn`, using column transformers for preprocessing.

**Evaluation Metrics:**

* Accuracy
* Precision
* Recall
* F1 Score

>  **Best performing model: Gradient Boosting Classifier**
>
> * Accuracy: **91.96%**
> * Good balance between bias and variance
> * Outperformed Logistic Regression, Random Forest, and KNN in F1 score and Recall.

---

##  Tech Stack

| Component      | Technology            |
| -------------- | --------------------- |
| Language       | Python 3.10           |
| Model Training | scikit-learn          |
| Web Framework  | Flask                 |
| Frontend       | Bootstrap 5           |
| Deployment     | GitHub + Render/Local |

---

##  About the App

This Flask web app allows users to input personal and lifestyle data through a friendly form. Based on this, it returns:

* A probability score
* A risk category: **Low**, **Moderate**, or **High**
* A suggestion to consult professionals (if needed)

**Use Cases:**

* Universities for mental health monitoring
* Startups or HR platforms for employee wellness
* Self-assessment tool for individuals

---

##  Getting Started

**To run locally:**

```bash
git clone https://github.com/your-username/mental-health-prediction.git
cd mental-health-prediction
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.


---

