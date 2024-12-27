# Evaluation script
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt
from models import BayesianMNISTCNN
from data_utils import get_data_loaders
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print(f"youre on Cuda {torch.cuda}")
elif torch.backends.mps.is_available():
    device = "mps"

# Load the trained model
def load_model(save_dir):
    deep_ensemble = []
    # in case the training is already done
    for i in range(8):
        path = os.path.join(save_dir, f"trained_model_{i}.pt")
        model = BayesianMNISTCNN()

        model.load_state_dict(torch.load(path))
        deep_ensemble.append(model)
    return deep_ensemble

# A function for predicting log probabilities with a deep ensemble (eval mode and averaging over the ensemble predictions)
@torch.inference_mode()
def deep_ensembles_predict_log_probs(models: list[BayesianMNISTCNN], data_loader: DataLoader, device='cuda') -> torch.Tensor:
    """Predict log probabilities with Deep Ensembles.

    Args:
        models: List of models
        data_loader: DataLoader for the data to predict
        device: Device to use for prediction
    Returns:
        Tensor of shape (n_models, n_inputs, n_classes) containing log_softmax outputs
    """
    for model in models:
        model.to(device)
        model.eval()
    all_outputs = []
    for data, _ in tqdm(data_loader, desc="Predicting with Deep Ensembles"):
        data = data.to(device)
        outputs = []
        for model in models:
            log_probs, _ = model(data)
            outputs.append(log_probs)
        all_outputs.append(torch.stack(outputs, dim=0))
    for model in models:
        model.to('cpu')
    return torch.cat(all_outputs, dim=1)

# Evaluating the deep ensemble on MNIST’s test set in terms of NLL and accuracy
def compute_performance_metrics(log_probs: torch.Tensor, data_loader: DataLoader) -> tuple[float, float]:
    """Compute accuracy and negative log likelihood metrics.

    Args:
        log_probs: Tensor of shape (n_samples, n_inputs, n_classes) containing log_softmax outputs
        data_loader: DataLoader containing the data and labels

    Returns:
        Tuple of (accuracy, negative log likelihood)
    """
    device = log_probs.device

    # Average predictions over samples
    mean_log_probs = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(log_probs.shape[0], device=device))

    all_labels = []
    for _, labels in data_loader:
        all_labels.append(labels)
    labels = torch.cat(all_labels).to(device)

    # Compute accuracy
    predictions = mean_log_probs.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item()

    # Compute negative log likelihood
    nll = -mean_log_probs[range(len(labels)), labels].mean().item()

    return accuracy, nll

# Using one of the trained models to predict log probabilities with MC Dropout (using train mode to get different predictions for each sample).
def mc_dropout(mc_model, data_loader, n=32, device='cuda'):
    all_outputs = []

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc="Predicting with MC Dropout Ensembles"):
            data = data.to(device)
            outputs = []
            for iteration in range(n):
                log_probs, _ = mc_model(data)
                outputs.append(log_probs)
            all_outputs.append(torch.stack(outputs, dim=1))
    all_outputs = torch.cat(all_outputs, dim=0)

    return all_outputs.permute(1, 0, 2)  # Shape: (32, total_batch_size, num_classes)

# Additional evaluation functions for uncertainty computation
def compute_accuracy_metric(predictive_log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for given predictions and labels."""
    predictions = predictive_log_probs.argmax(dim=1)
    return (predictions == labels).float().mean().item()

def compute_nll_metric(predictive_log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute negative log likelihood for given predictions and labels."""
    return -predictive_log_probs[range(len(labels)), labels].mean().item()


@torch.inference_mode()
def sample_predictions(model, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
    """Get samples from the predictive distribution.

    Args:
        model: Neural network with dropout/sampling capability
        x: Input tensor of shape (batch_size, ...)
        n_samples: Number of Monte Carlo samples

    Returns:
        Tensor of shape (n_samples, batch_size, n_classes) containing log_softmax outputs
    """
    model.train()  # Enable dropout
    samples = torch.stack([
        model(x)  # Assuming model outputs log_softmax
        for _ in range(n_samples)
    ])
    model.eval()
    return samples

@torch.inference_mode()
def compute_predictive_distribution(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute mean predictive distribution from samples.

    Args:
        log_probs: Tensor of shape (n_samples, batch_size, n_classes)

    Returns:
        Mean probabilities of shape (batch_size, n_classes)
    """
    return torch.exp(log_probs).mean(dim=0)

@torch.inference_mode()
def torch_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distributions.

    Args:
        probs: Tensor of shape (..., n_classes)

    Returns:
        Entropy of each distribution
    """
    entropy = -probs * torch.log(probs)
    # Replace NaNs (from 0 * log(0)) with 0
    entropy = torch.nan_to_num(entropy, nan=0.0)
    return entropy.sum(dim=-1)

@torch.inference_mode()
def compute_uncertainties(log_probs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Compute different uncertainty measures from samples.

    Args:
        log_probs: Tensor of shape (n_samples, batch_size, n_classes)

    Returns:
        predictive_distribution: Mean probabilities across samples
        total_uncertainty: Predictive entropy
        aleatoric_uncertainty: Expected entropy of individual predictions
        epistemic_uncertainty: Mutual information (total - aleatoric)
    """
    # Get probabilities from log_softmax
    probs = torch.exp(log_probs)

    # Compute predictive distribution
    pred_dist = probs.mean(dim=0)

    # Compute uncertainties
    total_uncertainty = torch_entropy(pred_dist)
    aleatoric_uncertainty = torch_entropy(probs).mean(dim=0)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return (
        pred_dist,
        total_uncertainty,
        aleatoric_uncertainty,
        epistemic_uncertainty
    )

# Main script for evaluation
if __name__ == '__main__':
    save_dir = "./deep_ensemble_trained"
    deep_ensemble = load_model(save_dir)
    train_loader, test_loader_mnist, test_loader_dirty_mnist, test_loader_fashion = get_data_loaders(batch_size=64, eval_batch_size=1024, device="cuda")

    # For the Deep Ensemble
    ensemble_log_probs = deep_ensembles_predict_log_probs(deep_ensemble, test_loader_mnist, device='cuda')
    ensemble_accuracy, ensemble_nll = compute_performance_metrics(ensemble_log_probs, test_loader_mnist)
    print(f"The Deep Ensemble Accuracy is {ensemble_accuracy:.2f}, and the Negative Log Likelihood is {ensemble_nll:.2f}")

    # For the Deterministic Model
    Deterministic = [deep_ensemble[0]]
    log_probs_deterministic = deep_ensembles_predict_log_probs(Deterministic, test_loader_mnist, device='cuda')
    Deterministic_accuracy, Deterministic_nll = compute_performance_metrics(log_probs_deterministic, test_loader_mnist)
    print(f"The Deterministic Model Accuracy is {Deterministic_accuracy:.2f}, and the Negative Log Likelihood is {Deterministic_nll:.2f}")

    # For MC Dropout BNN
    MC_DROPOUT_MODEL = deep_ensemble[0]
    MC_DROPOUT_MODEL.to(device)
    MC_DROPOUT_MODEL.train()

    MC_log_probs = mc_dropout(MC_DROPOUT_MODEL, test_loader_mnist)
    MC_accuracy, MC_nll = compute_performance_metrics(MC_log_probs, test_loader_mnist)
    print(f"The Accuracy for the MC Dropout Model is {MC_accuracy:.3f}, and the NLL is {MC_nll:.3f}")

    # Additional Uncertainty Analysis on Dirty-MNIST
    targets = torch.cat([target for _, target in test_loader_dirty_mnist], dim=0)
    log_probs_deep_ensemble = deep_ensembles_predict_log_probs(deep_ensemble, test_loader_dirty_mnist, device='cuda')
    pred_dist_de, total_uncertainty_de, aleatoric_uncertainty, epistemic_uncertainty = compute_uncertainties(log_probs_deep_ensemble)
    deep_ensemble_method = {
        "name": "Deep Ensemble",
        "total_uncertainty": total_uncertainty_de,
        "pred_dist": pred_dist_de,
    }

    # MC Dropout Uncertainty Analysis
    log_probs_mc = mc_dropout(MC_DROPOUT_MODEL, test_loader_dirty_mnist)
    pred_dist_mc, total_uncertainty_mc, aleatoric_uncertainty, epistemic_uncertainty = compute_uncertainties(log_probs_mc)
    MC_method = {
        "name": "MC Dropout",
        "total_uncertainty": total_uncertainty_mc,
        "pred_dist": pred_dist_mc,
    }

    # Deterministic Model Uncertainty Analysis
    log_probs_det_model = deep_ensembles_predict_log_probs([deep_ensemble[0]], test_loader_dirty_mnist, device='cuda')
    pred_dist_det, total_uncertainty_det, aleatoric_uncertainty, epistemic_uncertainty = compute_uncertainties(log_probs_det_model)
    det_model_method = {
        "name": "Deterministic",
        "total_uncertainty": total_uncertainty_det,
        "pred_dist": pred_dist_det,
    }

    # Save methods for plotting
    methods = [det_model_method, MC_method, deep_ensemble_method]
    with open("methods.pkl", "wb") as f:
        pickle.dump(methods, f)
    print("Methods saved to methods.pkl")

    with open("targets.pkl") as ff :
        pickle.dump(targets,ff)
    print("Targets saved to targets.pkl for later visualization")
