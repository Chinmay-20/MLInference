# MLInference

A hybrid **Python + C++ machine learning inference system** that trains a PyTorch model, exports it to ONNX, and serves predictions through a native **ONNX Runtime C++ backend** exposed via **FastAPI**.

This project started from an ML-to-C++ deployment demo and was extended into a more production-style inference service with:

- dynamic batch ONNX export
- JSON-based native C++ inference
- FastAPI endpoints for online prediction
- request validation
- latency benchmarking
- automated API integration tests

---

## Architecture

```text
Python Training (PyTorch + Hydra + MLflow)
        |
        v
   ONNX Export
        |
        v
C++ Inference Worker (ONNX Runtime)
        |
        v
 FastAPI Service Layer
        |
        v
 REST Endpoints + Benchmarking + Tests

Main components
    PyTorch training pipeline: trains a small regression model on synthetic data
    ONNX export: exports the trained model with dynamic batch support
    C++ inference worker: loads the ONNX model and performs inference using ONNX Runtime
    FastAPI service: exposes the inference worker through HTTP endpoints
    Benchmark script: measures latency and throughput
    Integration tests: validates the API end to end

## Project Structure
MLInference/
├── conf/                     # Hydra configuration
├── cpp_inference/            # Native C++ inference worker
│   ├── config.yaml
│   ├── CMakeLists.txt
│   └── src/main.cpp
├── data/                     # Generated dataset
├── docker/                   # Dockerfiles
├── ml_src/                   # Python ML training and export code
│   ├── data_loader.py
│   ├── export_onnx.py
│   ├── model.py
│   └── train.py
├── models/                   # Saved PyTorch and ONNX models
├── scripts/                  # Utility scripts
│   └── benchmark_api.py
├── service/                  # FastAPI inference service
│   └── app.py
├── tests/                    # API integration tests
│   └── test_service_api.py
├── requirements.txt
└── README.md

## Features
    Train a PyTorch model with Hydra-configured settings
    Track training artifacts with MLflow
    Export ONNX model with dynamic batch dimension
    Run inference from a native C++ binary using ONNX Runtime
    Accept JSON input through stdin or file for C++ inference
    Serve predictions over HTTP using FastAPI
    Support both single prediction and batch prediction
    Benchmark latency and throughput
    Validate API behavior with automated tests

## Tech Stack
    Python 3.13
    PyTorch
    Hydra
    MLflow
    ONNX
    ONNX Runtime
    C++17
    FastAPI
    Pytest
    CMake
    yaml-cpp

## Python Version Requirement

Use Python 3.13 for this project.

Hydra has compatibility issues with Python 3.14 in this setup, so training may fail if you create the virtual environment with Python 3.14.

Check your version:
python3.13 --version

## Setup
### 1. Clone the repository
git clone https://github.com/Chinmay-20/MLInference.git
cd MLInference

### 2. Create a virtual environment with Python 3.13
python3.13 -m venv venv_ml_inference
source venv_ml_inference/bin/activate
python -V

### 3. Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

### Train the Model and Export ONNX

From the repo root:
python ml_src/train.py

Expected outputs:
    trained PyTorch model saved in models/model.pth
    ONNX model saved in models/model.onnx

## Build the C++ Inference Worker
### Native dependencies on macOS
brew install onnxruntime yaml-cpp cmake

## Build commands
    cd cpp_inference
    rm -rf build
    mkdir build
    cd build

    cmake .. \
    -DONNXRUNTIME_DIR=/opt/homebrew/opt/onnxruntime \
    -DCMAKE_PREFIX_PATH="$(brew --prefix yaml-cpp);/opt/homebrew/opt/onnxruntime"

    cmake --build . -j$(sysctl -n hw.ncpu)

### Single input

echo '{"inputs": [[1,2,3,4,5,6,7,8,9,10]]}' | ./onnx_inference --stdin

### Batch input

echo '{"inputs": [[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}' | ./onnx_inference --stdin

### Run the FastAPI Service

From the root repo
uvicorn service.app:app --reload --host 127.0.0.1 --port 8000

if port 8000 is already in use
uvicorn service.app:app --reload --host 127.0.0.1 --port 8000

### API Endpoints
curl http://127.0.0.1:8000/

### Health check
curl http://127.0.0.1:8000/health

### Single Prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"input":[1,2,3,4,5,6,7,8,9,10]}'

### Batch Prediction
curl -X POST "http://127.0.0.1:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{"inputs":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'

### Benchmarking
python scripts/benchmark_api.py

### API integration tests
pytest tests/test_service_api.py -v

## Verified Workflow

The following end-to-end flow has been verified locally:

    Train PyTorch model
    Export ONNX model with dynamic batch support
    Build native C++ inference worker
    Run FastAPI service
    Execute single and batch predictions successfully
    Measure latency with benchmark script
    Validate endpoints with automated tests

Why this project matters

This project demonstrates the bridge between:
    ML experimentation in Python
    and native inference execution in C++

It shows how to move from a training workflow into a more deployment-oriented system with:
    reproducible model export
    native runtime integration
    service exposure
    request validation
    performance measurement
    automated testing

