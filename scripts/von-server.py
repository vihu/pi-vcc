"""Von server for pi-vcc's `profile: "von"`. Same /v1/systemone wire protocol as `von serve`, two differences:

1. Every question in a request is packed into one padded batch and scored in a few
   forward passes instead of one pass per question.
2. A question whose instructions reference `candidates[k]` is evaluated against
   {…state without candidates, candidate: candidates[k]} with the reference rewritten
   to `candidate`, i.e. fan-out over an array state.

Noul/choice/score math, the null-pass debias for criteria-less nouls, and the
input-conditioned temperature are the same as von.backends.option_marker_backend.

Setup (GPU wheel index for your machine: rocm7.14, cu128, or cpu):
  uv venv .venv-von && uv pip install --python .venv-von/bin/python torch --index-url https://download.pytorch.org/whl/rocm7.14
  uv pip install --python .venv-von/bin/python von-sdk fastapi uvicorn
  VON_DTYPE=bf16 .venv-von/bin/python -m uvicorn --app-dir scripts von-server:app --port 8000
Then in pi-vcc-config.json: "localModel": { "enabled": true, "url": "http://localhost:8000", "profile": "von" }.
VON_DEVICE (default cuda; ROCm also uses "cuda"), VON_DTYPE (fp32 | bf16 | fp16), VON_BATCH (32).
"""
import math
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from von.backends.option_marker_backend import OptionMarkerBackend, _format_state

backend = OptionMarkerBackend(device=os.environ.get("VON_DEVICE", "cuda"))
model = backend._get_model()
tok = model.tokenizer
BATCH = int(os.environ.get("VON_BATCH", "32"))
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(os.environ.get("VON_DTYPE", "fp32"))
if DTYPE is not None:
    model.encoder.to(DTYPE)
MAX_LEN = int(os.environ.get("VON_MAX_LEN", "8192"))
REF = re.compile(r"`candidates\[(\d+)\]`")
app = FastAPI()
_null_cache: Dict[str, torch.Tensor] = {}


class Req(BaseModel):
    model: str = "von-latest"
    state: Any
    questions: Dict[str, Dict[str, Any]]


def state_view(state: Any, instr: str) -> Tuple[str, str]:
    m = REF.search(instr) if isinstance(instr, str) else None
    if m and isinstance(state, dict) and isinstance(state.get("candidates"), list):
        view = {k: v for k, v in state.items() if k != "candidates"}
        view["candidate"] = state["candidates"][int(m.group(1))]
        return _format_state(view), instr.replace(m.group(0), "`candidate`")
    return _format_state(state), instr if isinstance(instr, str) else _format_state(instr)


def options_of(q: Dict[str, Any]) -> Tuple[str, List[str], List[str], bool]:
    t = q.get("type", "noul")
    crit = q.get("criteria") or {}
    if t == "noul":
        explicit = bool(crit.get("true") or crit.get("false"))
        return t, ["true", "false"], [crit.get("true") or "Yes, condition holds true.", crit.get("false") or "No, condition is false."], explicit
    if t == "choice":
        keys = list(crit.keys())
        return t, keys, [str(crit[k]).strip() if crit[k] else k for k in keys], True
    if t == "score":
        descs = [(d.get("what", "") if isinstance(d, dict) else str(d)).strip() for d in crit]
        return t, [str(i) for i in range(len(descs))], descs, True
    raise ValueError(f"unknown question type {t}")


def run_batch(packed: List[str]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for start in range(0, len(packed), BATCH):
        enc = tok(packed[start:start + BATCH], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(backend.device)
        positions = [(enc["input_ids"][b] == model.mask_token_id).nonzero(as_tuple=True)[0].tolist() for b in range(enc["input_ids"].shape[0])]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE or torch.float32, enabled=DTYPE is not None):
            out.extend(l.detach().float().cpu() for l in model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], mask_positions=positions))
    return out


def null_logits(instr: str, descs: List[str]) -> torch.Tensor:
    key = instr + "\x00" + "\x00".join(descs)
    if key not in _null_cache:
        _null_cache[key] = run_batch([model.pack_sequence("", instr, descs)])[0]
    return _null_cache[key]


@app.get("/health")
def health():
    return {"status": "ok", "service": "von-batch", "batch": BATCH}


@app.post("/v1/systemone")
def system_one(req: Req):
    items = []
    for qid, q in req.questions.items():
        st, instr = state_view(req.state, q.get("instructions", ""))
        t, keys, descs, explicit = options_of(q)
        items.append((qid, t, keys, descs, explicit, st, instr, model.pack_sequence(st, instr, descs)))
    logits_all = run_batch([it[7] for it in items])
    answers: Dict[str, Any] = {}
    for (qid, t, keys, descs, explicit, st, instr, _), logits in zip(items, logits_all):
        if t == "noul" and not explicit:
            nl = null_logits(instr, descs)
            bias = nl[0] - nl[1]
            logits = torch.stack([logits[0] - 0.7 * bias, logits[1]])
        temp = backend._effective_temperature(logits, st, len(descs), tok, None)
        probs = torch.softmax(logits / max(temp, 1e-4), dim=-1).tolist()
        sp = sorted(probs, reverse=True)
        conf = round(max(0.0, min(1.0, sp[0] - (sp[1] if len(sp) > 1 else 0.0))), 3)
        if t == "noul":
            answers[qid] = {"type": "noul", "noul": round(max(0.0, min(1.0, probs[0])), 4)}
        elif t == "choice":
            answers[qid] = {"type": "choice", "choice": keys[int(torch.argmax(logits))],
                            "probabilities": {k: round(p, 4) for k, p in zip(keys, probs)}, "confidence": conf}
        else:
            answers[qid] = {"type": "score", "score": round(sum(i * p for i, p in enumerate(probs)), 2), "confidence": conf,
                            "legend": dict(zip(keys, descs)), "probabilities": {k: round(p, 4) for k, p in zip(keys, probs)}}
    return {"model": "von-1.1.0-batch", "answers": answers, "usage": {"input_tokens": sum(len(it[7]) // 4 for it in items), "output_tokens": len(answers)}}
