# Smart - Student Status AI Predictor

This is a simple Streamlit demo app that uses a machine learning model to predict a student's academic status (e.g., Active, On Probation, Deferred) based on performance data.

## 🚀 Features
- Synthetic dataset generation for quick testing
- Random Forest classifier for prediction
- Probability breakdown with visualization
- Easy-to-use Streamlit UI

## 📦 Installation
Clone this project or download the `smart_streamlit_demo.py` file.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the App
Run the Streamlit app locally:

```bash
streamlit run smart_streamlit_demo.py
```

Then open the local URL printed in the terminal (usually `http://localhost:8501`).

## ☁️ Deploy on Streamlit Cloud
1. Push the project (this `.py` file and `requirements.txt`) to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Connect your GitHub repo and deploy the app.

## 📊 Usage
- Adjust student performance parameters from the left sidebar.
- Click **Predict** to see the predicted status and probability breakdown.
- View sample data and model performance at the bottom of the page.

---

⚠️ **Note:** This demo uses a synthetic dataset and a basic model for demonstration purposes. For real-world use, replace the dataset with real data and retrain the model.
