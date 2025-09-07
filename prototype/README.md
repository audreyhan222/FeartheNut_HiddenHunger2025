# NutriScope Web App

A Flask web application for Hidden Hunger risk assessment.

## Prerequisites
- Python 3.11 (recommended)
- pip
- (Optional) virtualenv

## 1) Create and activate a virtual environment
```bash
cd /Users/audre/code/personal/AppDesign_FeartheNut
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
# /Users/audre/code/personal/AppDesign_FeartheNut/DemoAPP/.env
SECRET_KEY=change-this
# Optional: required if you enable AI calls
GOOGLE_API_KEY=your_google_genai_api_key
```

## 4) Ensure instance folder exists (for SQLite)
Flask/SQLAlchemy will place the SQLite DB in the `instance/` folder.
```bash
mkdir -p /Users/audre/code/personal/AppDesign_FeartheNut/DemoAPP/instance
```

## 5) Initialize the database
If you need to (first run or after deleting the DB), run:
```bash
cd /Users/audre/code/personal/AppDesign_FeartheNut/DemoAPP
python init_db.py
```
This will create the SQLite database under `DemoAPP/instance/`.

## 6) Model file (ML prediction)
If you are using the risk prediction route, ensure the model file exists at:
```
/Users/audre/code/personal/AppDesign_FeartheNut/DemoAPP/routes/my_model.sav
```
`routes/model.py` loads the model from its own directory.

## 7) Run the app
```bash
cd /Users/audre/code/personal/AppDesign_FeartheNut/DemoAPP
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

