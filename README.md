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
   git clone <repository-url>
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