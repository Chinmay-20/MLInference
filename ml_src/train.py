import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig        # type of configuration Hydra

from model import SimpleMLP
from data_loader import get_dataloaders
from export_onnx import export_model_to_onnx

from hydra.utils import get_original_cwd

import mlflow
import mlflow.pytorch


'''
Adjustable training with Hydra

1) Read YAML configuration
2) Create model
3) Load / generate data
4) Train
5) Store model
6) Track experiment with MLflow
'''

# Hydra act here, specifing the config_path ../conf/config.yaml, convert in cfg object
# pass it in train function
# In practice: cfg = load_yaml("conf/config.yaml") -> train(cfg)
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):

    # ================== MLflow Setup ==================
    mlflow.set_experiment(cfg.tracking.experiment_name)

    # Everything inside this block will be tracked
    with mlflow.start_run():

        # Save full Hydra configuration for reproducibility
        mlflow.log_text(str(cfg), "config.yaml")

        # Log main hyperparameters
        mlflow.log_params({
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "epochs": cfg.training.epochs,
            "input_size": cfg.model.input_size,
            "hidden_size": cfg.model.hidden_size,
            "seed": cfg.training.seed,
        })

        print(f"--- Training Configuration ---")
        print(f"Device: {cfg.training.device}")
        print(f"Epochs: {cfg.training.epochs}")
        print(f"Learning Rate: {cfg.training.learning_rate}")
        print(f"SEED: {cfg.training.seed}")

        SEED = cfg.training.seed
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # ================== Setup Device (Agnostic Hardware) ==================
        if cfg.training.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(cfg.training.device)

        print(f"Using device: {device}")

        # ========= Initialize Model with params from config.yaml ========
        model = SimpleMLP(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            output_size=cfg.model.output_size
        ).to(device)

        # ====== Load Data =========
        '''
        - Load data if exists
        - Create dataset if not
        - Automatic batching
        - Automatic shuffle
        '''
        dataloader = get_dataloaders(cfg)

        # ======= Loss and Optimizer =========
        criterion = nn.MSELoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate
        )

        # ================== Training Loop ==================
        model.train()

        for epoch in range(cfg.training.epochs):

            epoch_loss = 0.0

            for inputs, targets in dataloader:

                # move batch to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            print(f"Epoch [{epoch+1}/{cfg.training.epochs}] - Loss: {avg_loss:.4f}")

            # Log metric to MLflow
            mlflow.log_metric("loss", avg_loss, step=epoch)

        # ======= Save Model Locally ==========
        # Use an absolute path to save the model
        save_full_path = os.path.join(
            get_original_cwd(),
            cfg.paths.model_save
        )

        os.makedirs(os.path.dirname(save_full_path), exist_ok=True)
        torch.save(model.state_dict(), save_full_path)

        print(f"Model saved to {save_full_path}")

        # ======= Log PyTorch Model to MLflow ==========
        mlflow.pytorch.log_model(model, "model")

        # ===== Export ONNX ========
        onnx_full_path = os.path.join(
            get_original_cwd(),
            cfg.paths.model_onnx
        )

        os.makedirs(os.path.dirname(onnx_full_path), exist_ok=True)

        export_model_to_onnx(
            model=model,
            output_path=onnx_full_path,
            input_size=cfg.model.input_size,
            device=device
        )

        # ======= Log ONNX artifact to MLflow ==========
        mlflow.log_artifact(onnx_full_path, artifact_path="onnx_model")

        print(f"ONNX model saved and logged to MLflow")


if __name__ == "__main__":
    train()
