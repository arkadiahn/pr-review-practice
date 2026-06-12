# ml-pipeline-bugfix

A Spotify Top 100 ML pipeline built with Dagster, FastAPI, and LakeFS.  
This repo is intentionally broken — your job is to find the bugs, fix them, and ship clean PRs.

---

## Stack

| Layer | Technology |
|---|---|
| Pipeline orchestration | Dagster |
| Feature engineering + training | scikit-learn / pandas |
| Model serving | FastAPI |
| Data versioning | LakeFS |

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
export DATA_DIR=data/
export MODEL_PATH=model.pkl
export MODEL_REGISTRY_PATH=/tmp/model_registry

# LakeFS (not required to find the bugs, but needed to run the full pipeline)
export LAKEFS_ENDPOINT=http://localhost:8000
export LAKEFS_ACCESS_KEY=your_access_key
export LAKEFS_SECRET_KEY=your_secret_key
export LAKEFS_REPO=spotify-pipeline
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Launch Dagster UI

```bash
dagster dev -f dagster_pipeline/assets.py
```

### 5. Start the API server

```bash
uvicorn api.main:app --reload
```

---

## Repo structure

```
ml-pipeline-bugfix/
├── dagster_pipeline/
│   ├── assets.py        # Pipeline assets: ingest → features → train → evaluate
│   ├── partitions.py    # Monthly partition definition
│   └── resources.py     # LakeFS + model registry Dagster resources
├── api/
│   └── main.py          # FastAPI prediction service
├── ml/
│   ├── features.py      # Feature engineering
│   └── model.py         # Training, evaluation, serialization
├── tests/
│   └── test_features.py # Unit tests for feature engineering
├── lakefs_config.yml    # LakeFS branch/path configuration
└── requirements.txt
```

---

## Workflow

1. Pick a bug at the standup — self-assign so there are no conflicts.
2. Create a branch with a descriptive name: `fix/<short-description>`
3. Make your fix with a meaningful commit message (what broke + why).
4. Open a PR with:
   - A clear title
   - A description explaining the bug and your fix
   - Any relevant context (e.g. what the impact was)
5. Assign a peer reviewer — they should leave at least one comment before approving.
6. Reviewer approves → merge.

---

## Data format

Each monthly CSV in `data/` should have these columns:

| Column | Type | Description |
|---|---|---|
| `track_id` | str | Spotify track ID |
| `track_name` | str | Track title |
| `artist_name` | str | Artist name |
| `track_date` | date | Date the track charted |
| `popularity` | int (0–100) | Target variable |
| `danceability` | float (0–1) | |
| `energy` | float (0–1) | |
| `loudness` | float (≤0 dB) | |
| `speechiness` | float (0–1) | |
| `acousticness` | float (0–1) | |
| `instrumentalness` | float (0–1) | |
| `liveness` | float (0–1) | |
| `valence` | float (0–1) | |
| `tempo` | float | BPM |
| `duration_ms` | int | Track length in milliseconds |
| `explicit` | bool | |
| `key` | int (0–11) | |
| `mode` | int (0–1) | |
| `time_signature` | int (1–7) | |
