import onnx
import torch
import onnxruntime as ort
import numpy as np

from ml_src.export_onnx import export_model_to_onnx
from ml_src.model import SimpleMLP

def test_onnx_export_execution(tmp_path):
    """
    Smoke test for ONNX export.

    Verifies that:
    1) export function runs without crashing
    2) ONNX file is created
    3) ONNX file is valid
    """

    model = SimpleMLP(input_size=10, hidden_size=20, output_size=1) 
    
    # temporary path: <tmp>/model.onnx
    output_file = tmp_path / "test_model.onnx"

    # run export, using input_size as SimpleMLP class
    export_model_to_onnx(model=model, output_path=str(output_file), input_size=10)

    # check file exists
    assert output_file.exists(), "ONNX file was not created"

    # check ONNX model validity
    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)

    dummy_input = torch.randn(1, 10)

    # Verify Onnx inference
    ort_session = ort.InferenceSession(str(output_file))
    dummy_input_np = dummy_input.numpy()
    outputs = ort_session.run(None, {"x": dummy_input_np})

    print("Output shape:", outputs[0].shape)
    assert outputs[0].shape == (1, 1)  # Expected shape
