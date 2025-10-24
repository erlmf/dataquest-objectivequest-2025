# Predicting Criminal Sentence Duration from Legal Case Narratives (Dataquest 2025)

Hybrid NLP regression to predict **imprisonment duration (months)** from Indonesian court verdicts.  
Model = **TF-IDF + domain numeric + BERT embeddings + Target Encoding → LightGBM**, with **GroupKFold** and **linear calibration**.


---

## 📦 Dataset
Provided by Dataquest 2025 (Universitas Airlangga).  
- `train.csv` (16,573 rows): text + target `lama hukuman (bulan)`  
- `test.csv`  (6,666 rows): text only  
- `sample_submission.csv`

---

## 🧰 Setup

```bash
git clone https://github.com/erlmf/dataquest-objectivequest-2025.git
cd dataquest-objectivequest-2025
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

Minimal GPU not required (BERT embeddings can be precomputed or reduced via SVD).
pip install -r requirements.txt
