from pathlib import Path
import torch


def export_model_to_onnx(model, output_path: str, input_size: int, device="cpu"):
    """
    Export a trained PyTorch model to ONNX with dynamic batch dimension.
    """
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, input_size).to(device)

    print(f"Exporting model to {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["output"],
        dynamic_axes={
            "x": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    print("Model exported successfully.")
