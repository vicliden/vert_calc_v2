# Jump Analyzer — my small Streamlit app + analysis tool

This repo is my quick tool for estimating airtime and approximate jump height from mobile accelerometer CSVs (e.g. phyphox/EDA exports). I built the analysis code and included a Streamlit UI so you can upload files from your phone and get plots + metrics right away.

Quick start (local)
-------------------
1) Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run the Streamlit app from the repo root:

```powershell
streamlit run app/streamlit_app.py
```

4) Open the URL Streamlit prints (usually http://localhost:8501) in your phone or browser.

Running on Render
-----------------
I already included `render.yaml` so Render can pick it up. If you choose "Use existing render.yaml" when creating the service, it should be ready to go.

Start command used on Render:

```text
streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```
