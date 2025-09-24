
"""
Smart - Streamlit demo
This AI model predicts the academic status of students based on their performance data.
Uses synthetic data and trains a RandomForest classifier on app start (cached).
Requirements:
    pip install streamlit pandas scikit-learn matplotlib numpy
Run:
    streamlit run smart_streamlit_demo.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart", layout="centered")

# --- Header ---
st.title("Smart")
st.caption("This AI model predicts the academic status of students based on their performance data.")
st.write("A simple demo that trains on synthetic data and lets you make single-student predictions with probability breakdowns.")

# --- Data generation (synthetic) ---
@st.cache_data(show_spinner=False)
def generate_synthetic_student_data(n=1000, random_state=42):
    np.random.seed(random_state)
    departments = ['Computer Science','Business','Engineering','Arts']
    courses = ['CS101','MATH201','ENG150','BUS300']
    semesters = ['Sem 1','Sem 2']
    data = {
        'Student ID': range(1, n+1),
        'Full Name': [f'Student_{i}' for i in range(1, n+1)],
        'Department': np.random.choice(departments, n),
        'Course': np.random.choice(courses, n),
        'Semester': np.random.choice(semesters, n),
        'GPA': np.round(np.random.normal(2.8, 0.6, n).clip(0,4), 2),
        'Attendance %': np.round(np.random.normal(80, 10, n).clip(30,100), 1),
        'Exam Score': np.round(np.random.normal(65, 15, n).clip(0,100), 1),
        'Assignment Score': np.round(np.random.normal(70, 12, n).clip(0,100), 1),
    }
    df = pd.DataFrame(data)
    cond = (
        (df['GPA'] >= 3.0) & (df['Attendance %'] >= 75) & (df['Exam Score'] >= 60)
    )
    df['Status'] = np.where(cond, 'Active', 'On Probation')
    deferred_idx = df.sample(frac=0.05, random_state=random_state).index
    df.loc[deferred_idx, 'Status'] = 'Deferred'
    return df

df = generate_synthetic_student_data(1000)

# --- Model training (cached) ---
@st.cache_resource(show_spinner=False)
def train_model(df):
    features = ['Department','Course','Semester','GPA','Attendance %','Exam Score','Assignment Score']
    X = df[features].copy()
    y = df['Status'].copy()
    # encode categorical
    cat_cols = ['Department','Course','Semester']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])
        encoders[c] = le
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    # metrics
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'model': pipeline,
        'encoders': encoders,
        'features': features,
        'metrics': {'accuracy': acc, 'report': clf_report, 'confusion_matrix': cm},
        'classes': pipeline.classes_
    }

state = train_model(df)

# --- Sidebar: user inputs ---
st.sidebar.header("Enter student data")
dept = st.sidebar.selectbox("Department", sorted(df['Department'].unique()))
course = st.sidebar.selectbox("Course", sorted(df['Course'].unique()))
sem = st.sidebar.selectbox("Semester", sorted(df['Semester'].unique()))
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 2.5, 0.01)
att = st.sidebar.slider("Attendance %", 0.0, 100.0, 80.0, 0.1)
exam = st.sidebar.slider("Exam Score", 0.0, 100.0, 70.0, 0.1)
assn = st.sidebar.slider("Assignment Score", 0.0, 100.0, 75.0, 0.1)

if st.sidebar.button("Predict"):
    model = state['model']
    encs = state['encoders']
    feat = state['features']
    # prepare record
    rec = {'Department': dept, 'Course': course, 'Semester': sem,
           'GPA': gpa, 'Attendance %': att, 'Exam Score': exam, 'Assignment Score': assn}
    # encode
    for c in ['Department','Course','Semester']:
        rec[c] = encs[c].transform([rec[c]])[0]
    X_rec = pd.DataFrame([rec])[feat]
    pred = model.predict(X_rec)[0]
    proba = model.predict_proba(X_rec)[0]
    classes = model.classes_
    st.subheader("Prediction")
    st.success(f"Predicted Status: {pred}")
    # probabilities table
    prob_df = pd.DataFrame({'Class': classes, 'Probability': proba})
    prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
    st.write("Probability breakdown:")
    st.table(prob_df)

    # --- Matplotlib bar chart of probabilities ---
    fig, ax = plt.subplots()
    ax.bar(prob_df['Class'], prob_df['Probability'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)

# --- Show sample data and model metrics ---
st.markdown("---")
st.subheader("Sample of the synthetic dataset")
st.dataframe(df.sample(8, random_state=1).reset_index(drop=True))

st.subheader("Model performance on held-out test set")
st.write(f"Accuracy: {state['metrics']['accuracy']:.3f}")
# show classification report as table
report_df = pd.DataFrame(state['metrics']['report']).transpose()
st.dataframe(report_df)

st.write("Confusion matrix (rows: true, columns: predicted)")
cm = state['metrics']['confusion_matrix']
st.write(cm)

st.markdown("**Notes & next steps:**")
st.write("""
- The demo trains a basic RandomForest on synthetic data. For production, replace with your real CSV and persist the trained model.
- Consider using SHAP or feature importance to explain predictions.
- You can deploy this app on Streamlit Community Cloud: https://share.streamlit.io
""")
