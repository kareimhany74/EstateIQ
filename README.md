# EstateIQ — Deploy Ready

Files:
- main.py
- chatbot.py
- frontend.html
- real_estate_model.joblib
- requirements.txt
- render.yaml

Render:
Build Command:
    pip install -r requirements.txt

Start Command:
    uvicorn main:app --host 0.0.0.0 --port $PORT

Health Check:
    /health
