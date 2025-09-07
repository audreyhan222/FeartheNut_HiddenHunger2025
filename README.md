# Hidden Hunger Prediction Model

A machine learning project that predicts hidden hunger (micronutrient deficiency) using demographic and nutritional intake data. The project uses a Neural Network (MLP) classifier to identify individuals at risk of hidden hunger.

## Project Structure

```
FeartheNut_HiddenHunger2025/
├── code/
│   ├── data_cleaning.py      # Data preprocessing and cleaning functions
│   ├── model_building.py     # Main ML model training and evaluation
│   ├── hidden_hunger.csv     # Dataset
│   └── requirements.txt      # Python dependencies
├── outputs/                  # Model outputs and visualizations
├── App_Development/          # Application development files
└── README.md
```

## Dataset

The model uses a dataset containing:
- **Demographic features**: Age, Gender, Income_Bracket, Education_Level
- **Nutritional intake features**: Vitamin_A_Intake_ug, Vitamin_D_Intake_IU, Zinc_Intake_mg, Iron_Intake_mg, Folate_Intake_ug
- **Target variable**: Hidden_Hunger_Flag (binary classification)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/audreyhan222/FeartheNut_HiddenHunger2025.git
   cd FeartheNut_HiddenHunger2025
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r code/requirements.txt
   ```

## Running the Model

### Quick Start

To run the complete model training and evaluation pipeline:

```bash
cd code
python model_building.py
```

### Expected Output

When you run the script, you should see output similar to:

```
Test metrics:
  Accuracy: 0.XXX
  F1:       0.XXX
  ROC AUC:  0.XXX

Confusion matrix:
[[XX XX]
 [XX XX]]

Classification report:
              precision    recall  f1-score   support

           0       0.XXX     0.XXX     0.XXX        XX
           1       0.XXX     0.XXX     0.XXX        XX

    accuracy                           0.XXX        XX
   macro avg       0.XXX     0.XXX     0.XXX        XX
weighted avg       0.XXX     0.XXX     0.XXX        XX

Best F1 threshold (test set): 0.XXX
Metrics at optimized threshold:
  Accuracy: 0.XXX
  F1:       0.XXX
  ROC AUC:  0.XXX
```

# NutriScope Web App

A Flask web application for Hidden Hunger risk assessment.

## Prerequisites
- Python 3.11 (recommended)
- pip
- (Optional) virtualenv

## 1) Create and activate a virtual environment
```bash
cd ..
python -m venv myenv
source myenv/bin/activate
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

## 3) Environment variables
Create a `.env` file inside `DemoAPP/` (same folder as `main.py`). These are optional unless you use AI features.

```bash

# FeartheNut_HiddenHunger2025/prototype/AppDesign_FeartheNut/DemoAPP/.env
SECRET_KEY=change-this
# Optional: required if you enable AI calls
GOOGLE_API_KEY=your_google_genai_api_key
```

## 4) Ensure instance folder exists (for SQLite)
Flask/SQLAlchemy will place the SQLite DB in the `instance/` folder.
```bash
mkdir -p FeartheNut_HiddenHunger2025/prototype/AppDesign_FeartheNut/DemoAPP/instance
```

## 5) Initialize the database
If you need to (first run or after deleting the DB), run:
```bash
cd FeartheNut_HiddenHunger2025AppDesign_FeartheNut/DemoAPP
python init_db.py
```
This will create the SQLite database under `DemoAPP/instance/`.

## 6) Model file (ML prediction)
If you are using the risk prediction route, ensure the model file exists at:
```
FeartheNut_HiddenHunger2025/prototype/AppDesign_FeartheNut/DemoAPP/routes/my_model.sav
```
`routes/model.py` loads the model from its own directory.

## 7) Run the app
```bash
cd FeartheNut_HiddenHunger2025/prototype/AppDesign_FeartheNut/DemoAPP
python main.py
```
Then open `http://127.0.0.1:5000` (or `http://localhost:5000`).

## Project structure (high level)
```
AppDesign_FeartheNut/
├─ DemoAPP/
│  ├─ main.py                 # Flask app entry
│  ├─ models.py               # SQLAlchemy models
│  ├─ init_db.py              # DB initialization helper
│  ├─ routes/                 # Blueprints / routes
│  │  ├─ form.py, model.py, login.py, signup.py, ...
│  │  └─ my_model.sav         # ML model (expected here)
│  ├─ templates/              # HTML templates
│  └─ instance/               # SQLite DB lives here (created at runtime)
└─ requirements.txt


```
