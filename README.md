# Predicting Criminal Sentence Duration from Legal Case Narratives (Dataquest 2025)

Hybrid NLP regression to predict **imprisonment duration (months)** from Indonesian court verdicts.  
Model = **TF-IDF + domain numeric + BERT embeddings + Target Encoding → LightGBM**, with **GroupKFold** and **linear calibration**.

> **Leaderboard**: **17th / 109 (Public)** and **25th / 109 (Private)**.

---

## 📦 Dataset
Provided by Dataquest 2025 (Universitas Airlangga).  
- `train.csv` (16,573 rows): text + target `lama hukuman (bulan)`  
- `test.csv`  (6,666 rows): text only  
- `sample_submission.csv`

> Put them under `data/raw/`.

---

## 🏗️ Repo Structure
notebooks/ # exploratory & training notebooks
scripts/ # CLI scripts (clean → fe → train → calibrate → infer)
configs/ # default config (paths, params)
data/{raw,interim,processed}
models/ # saved LGBM models per fold (optional)

---

## 🧰 Setup

```bash
git clone https://github.com/<username>/indo-sentencing-nlp.git
cd indo-sentencing-nlp
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

Minimal GPU not required (BERT embeddings can be precomputed or reduced via SVD).
pip install -r requirements.txt
