# 🌸 Iris Classifier API

A production-ready machine learning API for Iris flower species classification, built with **FastAPI**, **scikit-learn**, **Bootstrap 5**, and **MLflow**.

---

## 📌 Features

* 🔍 Predict species of Iris flowers (Setosa, Versicolor, Virginica) using multiple models
* 🧠 Supports ensemble of models: LogisticRegression, KNN, SVM, DecisionTree, RandomForest, NaiveBayes, NeuralNetwork
* 🌐 RESTful API built with **FastAPI**, accepts JSON input and returns prediction probabilities
* 📊 Model training, hyperparameter tuning, and experiment tracking with **MLflow**
* ✅ Clean modular Python codebase with clear folder structure
* 🐳 Docker support (coming soon)

---

## 🛠️ Prerequisites

* Python 3.12+
* Dependencies listed in `requirements.txt`

---

## 🚀 Running the API locally

1. Clone the repository and install dependencies:

   ```bash
   git clone <repo-url>
   cd Iris_Project
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:

   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
   ```

3. Access the API at `http://127.0.0.1:8001`

---

## 📡 API Usage Example

Send a POST request with sepal and petal measurements:

```bash
curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"sepal":[7.4,2.8],"petal":[6.1,1.9]}' \
     http://127.0.0.1:8001/predict
```

### Sample Response:

```json
{
  "LogisticRegression": {"setosa":0.0,"versicolor":0.064,"virginica":0.936},
  "KNN": {"setosa":0.0,"versicolor":0.0,"virginica":1.0},
  "SVM": {"setosa":0.004,"versicolor":0.005,"virginica":0.991},
  "DecisionTree": {"setosa":0.0,"versicolor":0.0,"virginica":1.0},
  "RandomForest": {"setosa":0.0,"versicolor":0.0,"virginica":1.0},
  "NaiveBayes": {"setosa":0.0,"versicolor":0.0,"virginica":1.0},
  "NeuralNetwork": {"setosa":0.001,"versicolor":0.134,"virginica":0.866}
}
```

---

## ⚙️ Training Models

* Use `app/training.py` to train models.
* Configure training parameters in `app/config/model_config.yaml`.
* Use `experiment` flag in `training.py` to run model tuning with MLflow tracking.
* After tuning, best model parameters are saved in `app/config/model_best.yaml`.
* Run training with the `train` flag to fit and save models locally (`models/*.pkl`).

---

## 📁 Project Structure

```
Iris_Project
├── app
│   ├── config           # YAML config files for models
│   ├── logs             # Application logs
│   ├── main.py          # FastAPI application entry point
│   ├── model.py         # Model utilities (load/save)
│   ├── predict.py       # Prediction logic
│   ├── schemas.py       # Pydantic models
│   ├── training.py      # Training and tuning scripts
│   └── utils.py         # Helper functions
├── mlruns               # MLflow experiment tracking data
├── models               # Saved trained models (.pkl files)
├── notebook             # Exploratory data analysis notebooks
├── requirements.txt     # Python dependencies
├── static               # Bootstrap CSS/JS and images
└── templates            # HTML templates
```

---

## 🐳 Docker Support (Coming Soon)

Docker support is planned for easy deployment. The API will be containerized and published on Docker Hub soon.

---

## 🚧 Roadmap

* [x] Add multiple ML models and hyperparameter tuning
* [x] Connect models to FastAPI for prediction
* [x] Enable model tracking with MLflow
* [ ] Add Docker support for deployment
* [ ] Implement automated tests (unit and integration)

---