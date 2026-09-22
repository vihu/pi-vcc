"""Laya server for pi-vcc's `profile: "laya"`: a /v1/systemone endpoint (the same wire shape as scripts/von-server.py) over laya.Router.

Setup (GPU wheel index for your machine: rocm7.14, cu128, or cpu):
  uv venv .venv-laya && uv pip install --python .venv-laya/bin/python torch --index-url https://download.pytorch.org/whl/rocm7.14
  uv pip install --python .venv-laya/bin/python laya fastapi uvicorn
  .venv-laya/bin/python -m uvicorn --app-dir scripts laya-server:app --port 8000
Then in pi-vcc-config.json: "localModel": { "enabled": true, "url": "http://localhost:8000", "profile": "laya" }.
"""
import os
from typing import Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from laya import Router

router = Router(preload=True)
app = FastAPI()

class Req(BaseModel):
    model: str = "laya"
    state: Any
    questions: Dict[str, Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok", "service": "laya-shim"}

@app.post("/v1/systemone")
def system_one(req: Req):
    res = router.predict(req.state, req.questions, model=os.environ.get("LAYA_MODEL") or None)
    return {"model": res.get("model", "laya"), "answers": res["answers"], "usage": res.get("usage", {})}
