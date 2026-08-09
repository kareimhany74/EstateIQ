# EstateIQ — Property Intelligence Platform

EstateIQ is an end-to-end Egyptian real-estate valuation project: a trained XGBoost model, FastAPI service, interactive web interface, market dashboard, dynamic input constraints, and a guided property chatbot.

## Live demo

- Live demo: `https://YOUR-VERCEL-PROJECT.vercel.app`
- API documentation: `https://YOUR-VERCEL-PROJECT.vercel.app/docs`
- Source repository: `https://github.com/YOUR-USERNAME/EstateIQ`

Replace the placeholders above after deployment.

## Highlights

- Property price estimate in EGP with market-context evidence and confidence score.
- Clear **indicative valuation range** using the model's existing error calibration and local evidence strength. The range is explicitly presented as guidance, not as a guaranteed statistical coverage interval.
- Lightweight explainable AI using XGBoost's native `pred_contribs` values—no additional SHAP dependency. The API and interface show the strongest positive and negative contributors with human-readable labels.
- Anonymous in-memory usage analytics: prediction count, average estimate, most requested governorate, and most requested property type.
- Dynamic constraints prevent unsupported combinations before inference.
- Interactive training-data dashboard and conversational property assistant remain included.

## API endpoints

- `GET /` — web application
- `GET /health` — service and model status
- `GET /metadata` — supported categories and model metadata
- `GET /constraints` — context-sensitive input rules
- `GET /compounds` — compound search
- `GET /market-context` — local market baseline and evidence
- `POST /predict` — valuation, range, confidence, and explanation
- `GET /analytics` — anonymous process-local usage summary
- `POST /chat` — chatbot message
- `DELETE /chat/{session_id}` — reset a chat session

## Analytics scope

Usage analytics do not store request bodies, IP addresses, user identifiers, or chat sessions. They live only in the current Python process, so they reset after a restart, cold start, or scale-down. On serverless platforms, separate instances may each hold different counters. This is intentionally a lightweight portfolio/demo feature, not durable monitoring.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Open `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Deploy

### Vercel

Import the repository and keep the root directory at `./`. Vercel detects the FastAPI app in `main.py`. The project keeps the same deployment structure and adds no new dependency for explanations.

### Render

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## Main files

- `main.py` — FastAPI service, model inference, explanations, and usage analytics
- `chatbot.py` — conversational intake and valuation response formatting
- `frontend.html` — valuation UI, explanations, chatbot, and dashboards
- `real_estate_model.joblib` — trained model artifact and market maps
- `requirements.txt` — Python dependencies
- `render.yaml` — Render deployment configuration

## Notes

EstateIQ estimates are informational and are not financial advice, a formal appraisal, or a guaranteed sale price.
