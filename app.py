# app.py
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import os
from model_utils import load_artifacts, fe_single

app = FastAPI(title="Bank Fraud Anomaly API")
templates = Jinja2Templates(directory="templates")

# artifacts will be loaded on startup to avoid import-time side-effects
scaler = None
iso = None
autoencoder = None
feature_cols = None


@app.on_event("startup")
def load_models_on_startup():
    global scaler, iso, autoencoder, feature_cols
    scaler, iso, autoencoder, feature_cols = load_artifacts()

class TxIn(BaseModel):
    transaction_id: str = "tx_demo"
    customer_id: str = "cust_demo"
    amount: float = 10.0
    hour: int = 12
    is_international: int = 0
    channel: str = "mobile"
    merchant_cat: str = "grocery"

def score_transaction_dict(tx: dict):
    X_df = fe_single(tx, feature_cols)
    X_scaled = scaler.transform(X_df.values)
    # IsolationForest score -> higher = more anomalous (we invert decision_function)
    iso_score = -iso.decision_function(X_scaled)[0]
    # Autoencoder reconstruction error
    recon = autoencoder.predict(X_scaled)
    recon_err = float(np.mean((recon - X_scaled)**2))
    # simple combined score (normalized)
    # Normalization using simple min/max from training not available here; use logistic transform for demo
    combined = 0.6 * (iso_score) + 0.4 * (recon_err / (1.0 + recon_err))
    # Risk metadata heuristics
    amt = float(tx.get('amount', 0.0))
    # Base severity from combined (soft scale)
    combined_norm = 1.0 / (1.0 + np.exp(-combined))  # logistic to [0,1]
    # Add amount influence (cap at 1.0, 10000 USD considered very high)
    amount_influence = min(1.0, amt / 10000.0)
    severity = float(min(1.0, combined_norm * 0.6 + amount_influence * 0.4))

    # Risk level mapping
    if amt >= 10000:
        risk_level = "critical"
    elif amt >= 5000:
        risk_level = "high"
    elif severity > 0.75:
        risk_level = "high"
    elif severity > 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    # High-level recommendations based on risk
    recommendations = []
    if risk_level == 'critical':
        recommendations = [
            "Immediate block the transaction.",
            "Notify fraud operations and hold payout.",
            "Freeze related account until investigation completes.",
            "Collect full audit trail (IP, device, geolocation)."
        ]
    elif risk_level == 'high':
        recommendations = [
            "Flag for manual review and hold funds.",
            "Contact customer for verification.",
            "Check for prior suspicious history."
        ]
    elif risk_level == 'medium':
        recommendations = [
            "Soft alert the monitoring dashboard.",
            "Increase sampling for this customer."
        ]
    else:
        recommendations = ["No special action recommended."]
    # Explanation: top contributing pseudo-features (simple heuristics for demo)
    explanations = []
    if tx['amount'] > 1000:
        explanations.append("High transaction amount compared to typical.")
    if tx['is_international']:
        explanations.append("International transaction.")
    if tx['channel'] in ['web']:
        explanations.append("Online/web channel (higher risk in this demo).")
    if len(explanations) == 0:
        explanations.append("Unusual pattern detected by model.")
    return {
        "iso_score": float(iso_score),
        "recon_error": recon_err,
        "combined_score": float(combined),
        "explanations": explanations,
        "risk_level": risk_level,
        "severity": severity,
        "recommendations": recommendations,
        "amount_influence": amount_influence,
        "human_note": None,
    }

@app.post("/score")
async def score(tx: TxIn):
    d = tx.dict()
    result = score_transaction_dict(d)
    # return also a simple action suggestion
    action = "no_action"
    # Heuristic overrides: treat high-amount transactions as higher risk
    # If amount > 1000, take action regardless of combined_score
    try:
        amt = float(d.get('amount', 0))
    except Exception:
        amt = 0.0
    if amt > 1000:
        # stronger response for international or web transactions
        if int(d.get('is_international', 0)) or d.get('channel', '') == 'web':
            action = "block"
        else:
            action = "manual_review"
    else:
        # fallback to score-based decisions
        if result['combined_score'] > 1.5:
            action = "block"
        elif result['combined_score'] > 0.9:
            action = "manual_review"
        elif result['combined_score'] > 0.6:
            action = "soft_alert"
    return JSONResponse({"transaction": d, "result": result, "action": action})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    # Run without the code reloader to avoid import/reload side-effects
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
