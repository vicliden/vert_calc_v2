# Jump Analyzer — web UI (Streamlit)

This repository contains a small physics / accelerometer jump analysis library and a Streamlit web UI you can deploy to host a mobile-friendly interface.

Files added in this change:
- `app/streamlit_app.py` — Streamlit mobile-friendly uploader and UI that uses `analyze_jumps_from_folder` from `jump_analysis.py` and shows plots/metrics.
- `requirements.txt` — minimal dependencies for Streamlit + analysis.

Quick local run (developer machine):

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the printed URL in your phone browser on the same network.

Deploy to Render (short steps):
1. Push your repo to GitHub.
2. Create a new Web Service on Render.
3. Connect to your repo and branch.
4. Set Start Command:

```
streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

Render will use `requirements.txt` to build dependencies. After deploy you'll get a public URL that you can visit on your phone.

Privacy note: files you upload to a hosted service are processed on that host. If you want full local privacy, run the Streamlit app locally and access it from your phone on the same network.

If you want, I can also add a tiny example Colab notebook for instant cloud execution.
