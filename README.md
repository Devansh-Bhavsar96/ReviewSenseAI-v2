# ReviewSense AI v2

Streamlit app that analyzes laptop reviews (sentiment, battery / heating / display / performance signals),
compares laptops, predicts sentiment for text you type, and recommends laptops from a dated INR price snapshot.

## Quick start (Windows)

1. Open this folder in a terminal and create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
2. **Add your data** (see below), then run:
   ```powershell
   streamlit run app.py
   ```
   or just double-click `run.bat` (after step 1).

## Your data

The app reads `data/reviewsense_master.csv`. If that file is missing it falls back to
`data/sample_master.csv` (synthetic demo data, clearly labelled in the app).

To use your real data, copy your old project's `data/` folder contents into this project's `data/` folder.
If you only have the source files, rebuild the master file:
```powershell
python build_master_dataset.py
```
(`build_master_dataset.py` and `SOURCES.md` are unchanged copies from v1.)

## What changed from v1

- Navigation uses a simple native radio bar; no fragile CSS overrides. Light/dark mode now works through
  Streamlit's own theme (menu ⋮ → Settings), configured in `.streamlit/config.toml`.
- Removed calls that crash on many Streamlit versions (`alt=` in `plotly_chart`, `title=` in `st.info`);
  width handling falls back automatically on older/newer Streamlit.
- Logic moved to `core.py` (no Streamlit inside), UI stays in `app.py`.
- Much faster aspect analysis (one tokenization pass, exact-word matching: no more "fan" matching "fantastic"),
  with simple negation handling ("not good").
- Model is trained once, shows held-out accuracy, and is cached; no more disk model files.
- Compare no longer shows meaningless 0 price/battery columns; needs 2+ laptops.
- Reviews page: laptop search box, review counts in the dropdown, 50-row pages, instant CSV download.
- New "Try a review" page for live sentiment prediction.
- Recommendations: native cards (no raw HTML injection), formatted price/score columns, safe empty states.
- Falls back to a pure-Python Naive Bayes if scikit-learn can't load.

## Notes and limits

- Sentiment labels come from star ratings (4-5 positive, 3 neutral, 1-2 negative); the classifier is an educational baseline.
- Reviews with no known laptop name (the Parquet sentences) are skipped by the app, same as v1.
- Aspect detection is keyword-based and can miss sarcasm or synonyms.
- Prices are a dated snapshot; verify with the seller. No scraping or live APIs.

## Files

`app.py` UI · `core.py` logic · `make_sample_data.py` demo data · `build_master_dataset.py` data builder ·
`data/` datasets · `.streamlit/config.toml` theme · `requirements.txt`
