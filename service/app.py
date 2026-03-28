from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="ML HPC Inference Service", version="1.0.0")

REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_BUILD_DIR = REPO_ROOT / "cpp_inference" / "build"
CPP_BINARY = CPP_BUILD_DIR / "onnx_inference"


class PredictRequest(BaseModel):
    inputs: List[List[float]] = Field(..., min_length=1)

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("inputs cannot be empty")

        expected_len = len(value[0])
        if expected_len == 0:
            raise ValueError("each input row must contain at least one value")

        for i, row in enumerate(value):
            if len(row) != expected_len:
                raise ValueError(
                    f"all input rows must have same length; row 0 has length "
                    f"{expected_len}, row {i} has length {len(row)}"
                )
        return value


class PredictSingleRequest(BaseModel):
    input: List[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: List[float] | List[List[float]]
    batch_size: int
    output_dim: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    cpp_binary_exists: bool
    cpp_binary_path: str


def run_cpp_inference(payload: dict) -> dict:
    if not CPP_BINARY.exists():
        raise HTTPException(
            status_code=500,
            detail=f"C++ binary not found at {CPP_BINARY}. Build it first.",
        )

    start = time.perf_counter()

    try:
        result = subprocess.run(
            [str(CPP_BINARY), "--stdin"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=str(CPP_BUILD_DIR),
            check=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to execute C++ binary: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000.0

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "Unknown C++ inference error"
        raise HTTPException(status_code=500, detail=detail)

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse C++ response as JSON: {exc}. Raw output: {result.stdout}",
        )

    parsed["latency_ms"] = round(latency_ms, 3)
    return parsed


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        cpp_binary_exists=CPP_BINARY.exists(),
        cpp_binary_path=str(CPP_BINARY),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictSingleRequest) -> PredictResponse:
    payload = {"inputs": [req.input]}
    result = run_cpp_inference(payload)
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(req: PredictRequest) -> PredictResponse:
    result = run_cpp_inference(req.model_dump())
    return PredictResponse(**result)

@app.get("/")
def root():
    return {
        "message": "ML HPC Inference Service is running",
        "docs": "/docs",
        "health": "/health"
    }