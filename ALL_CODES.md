# DermaScan AI — Complete Code Reference

> All project source files documented. Last updated: May 4, 2026

## Project Structure

```
backend/
├── main.py              — FastAPI server (routes, ML inference, auth)
├── model.py             — EfficientNet-B0 model definition
├── train.py             — Training pipeline (Focal Loss, Mixup, OneCycleLR)
├── evaluate.py          — Model evaluation (confusion matrix, metrics)
├── database.py          — SQLAlchemy database setup
├── db_models.py         — User & ScanHistory ORM models
├── gradcam_utils.py     — Grad-CAM heatmap generator
├── requirements.txt     — Python dependencies
└── static/
    ├── index.html       — Full SPA (Auth, Dashboard, Scan, History, Profile)
    ├── style.css        — Complete design system
    └── app.js           — All frontend logic
```

## File Locations

All source files are located in: `e:\AI project\backend\`

| File | Path |
|------|------|
| Server | `e:\AI project\backend\main.py` |
| Model | `e:\AI project\backend\model.py` |
| Training | `e:\AI project\backend\train.py` |
| Evaluation | `e:\AI project\backend\evaluate.py` |
| Database | `e:\AI project\backend\database.py` |
| DB Models | `e:\AI project\backend\db_models.py` |
| Grad-CAM | `e:\AI project\backend\gradcam_utils.py` |
| HTML | `e:\AI project\backend\static\index.html` |
| CSS | `e:\AI project\backend\static\style.css` |
| JavaScript | `e:\AI project\backend\static\app.js` |
| Dependencies | `e:\AI project\backend\requirements.txt` |

## How to Run

```bash
cd "e:\AI project\backend"
pip install -r requirements.txt
python main.py
# Server starts at http://localhost:5000
```

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Serves index.html |
| POST | `/login` | Login → returns token |
| POST | `/register` | Create account |
| POST | `/predict` | Upload image → diagnosis |
| GET | `/history` | Scan history |
| GET | `/health` | Model status |

## Tech Stack

- **Backend**: FastAPI + Uvicorn (Python)
- **ML Model**: EfficientNet-B0 (PyTorch)
- **Database**: SQLite via SQLAlchemy
- **Frontend**: Vanilla HTML/CSS/JS (single-page app)
- **Heatmaps**: Grad-CAM (pytorch-grad-cam)
