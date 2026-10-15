"""
Web App Service (Root Gateway & Frontend)
Serves the Wise-grade UPI Fraud Detection Dashboard and coordinates with the
internal 'api' service using Vercel's multi-service binding.
"""
import os
import json
import urllib.request
import urllib.error
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(
    title="Sentinel UPI - Web App Service",
    description="Frontend gateway service with internal service binding to the GNN fraud detection API",
    version="1.0.0"
)

# Vercel service binding: Vercel injects the target service's base URL into API_URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """Serve the interactive Wise-grade dashboard."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "public", "index.html"),
        os.path.join(os.path.dirname(__file__), "dashboard", "index.html")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>Sentinel UPI Dashboard</h1>"


@app.get("/health")
def health_check():
    """Liveness check reporting status and bound internal API connection."""
    return {
        "status": "healthy",
        "service": "app",
        "bound_api_url": API_URL
    }


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def forward_to_internal_api(request: Request, path: str):
    """
    Proxies requests from the frontend or external clients to the internal
    'api' service via the injected Vercel service binding (API_URL).
    """
    target_url = f"{API_URL.rstrip('/')}/{path}"
    body = await request.body()
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(
        target_url,
        data=body if request.method == "POST" else None,
        headers=headers,
        method=request.method
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return JSONResponse(content=data, status_code=resp.status)
    except urllib.error.HTTPError as e:
        error_payload = e.read().decode("utf-8")
        try:
            return JSONResponse(content=json.loads(error_payload), status_code=e.code)
        except Exception:
            return JSONResponse(content={"error": error_payload}, status_code=e.code)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error connecting to internal 'api' service at {API_URL}: {str(e)}"
        )
