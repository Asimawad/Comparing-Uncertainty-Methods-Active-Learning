# -*- coding: utf-8 -*-
"""
OoD Detection and Density-Based Uncertainty
"""

# Imports
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.covariance import EmpiricalCovariance
from models import BayesianMNISTCNN
from data_utils import get_data_loaders

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained models and datasets
def load_models(save_dir="./deep_ensemble_trained"):
    """Load the trained ensemble models."""
    models = []
    for i in range(8):
        model = BayesianMNISTCNN()
        model.load_state_dict(torch.load(os.path.join(save_dir, f"trained_model_{i}.pt")))
        models.append(model)
    return models

# Load trained models
deep_ensemble = load_models()
train_loader, _, test_loader_dirty_mnist, test_loader_fashion = get_data_loaders()

# Function for OoD Detection ROC Curve
def plot_roc_curve(scores, labels, method_name):
    """Plot ROC curve for OoD detection."""
    fpr, tpr, _ = roc_curve(labels.cpu().numpy(), scores.cpu().numpy())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{method_name} (AUROC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for OoD Detection')
    plt.legend()

# Compute BALD for OoD Detection
@torch.no_grad()
def compute_BALD(models, data_loader, method="deep_ensemble"):
    if method == "deep_ensemble":
        log_probs = deep_ensembles_predict_log_probs(models, data_loader, device=device)
        pred_dist = torch.exp(log_probs).mean(dim=0)
        total_uncertainty = -torch.sum(pred_dist * torch.log(pred_dist + 1e-9), dim=1)
        aleatoric_uncertainty = -torch.sum(torch.exp(log_probs) * log_probs, dim=2).mean(dim=0)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
        return epistemic_uncertainty, total_uncertainty, aleatoric_uncertainty

# Predict log probabilities with deep ensemble
@torch.no_grad()
def deep_ensembles_predict_log_probs(models, data_loader, device='cuda'):
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
        model.to("cpu")
    return torch.cat(all_outputs, dim=1)

# Compute uncertainties for OoD Detection
epistemic_id, total_uncertainty_id, aleatoric_id = compute_BALD(deep_ensemble, test_loader_dirty_mnist)
epistemic_ood, total_uncertainty_ood, aleatoric_ood = compute_BALD(deep_ensemble, test_loader_fashion)

# Combine uncertainties and create labels
bald_scores = torch.cat([epistemic_id, epistemic_ood])
pred_ent_scores = torch.cat([total_uncertainty_id, total_uncertainty_ood])
cond_ent_scores = torch.cat([aleatoric_id, aleatoric_ood])
labels = torch.cat([torch.zeros_like(epistemic_id), torch.ones_like(epistemic_ood)])

# Plot OoD Detection ROC Curves
plt.figure(figsize=(8, 6))
plot_roc_curve(bald_scores, labels, "BALD")
plot_roc_curve(pred_ent_scores, labels, "Predictive Entropy")
plot_roc_curve(cond_ent_scores, labels, "Conditional Entropy")
plt.show()

# Density-Based Uncertainty Estimation
@torch.no_grad()
def extract_latents_and_labels(model, data_loader, device='cuda'):
    model.to(device)
    model.eval()
    latents, labels = [], []
    for inputs, targets in tqdm(data_loader, desc="Extracting latents"):
        inputs = inputs.to(device)
        _, latent = model(inputs)
        latents.append(latent.cpu())
        labels.append(targets.cpu())
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)

latents_train, labels_train = extract_latents_and_labels(deep_ensemble[0], train_loader)

def fit_class_gaussians(latents, labels):
    """Fit Gaussian distributions for each class."""
    class_gaussians = {}
    for class_id in range(10):  # Assuming 10 classes
        class_latents = latents[labels == class_id].numpy()
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(class_latents)
        class_gaussians[class_id] = cov_estimator
    return class_gaussians

class_gaussians = fit_class_gaussians(latents_train, labels_train)

def compute_mahalanobis(latents, class_gaussians):
    """Compute Mahalanobis distance for uncertainty."""
    uncertainties = []
    for latent in tqdm(latents.numpy(), desc="Computing Mahalanobis distance"):
        densities = []
        for class_id, gaussian in class_gaussians.items():
            dist = gaussian.mahalanobis(latent.reshape(1, -1))
            densities.append(-dist)
        uncertainties.append(np.max(densities))
    return np.array(uncertainties)

# Compute latents for test sets
latents_test_dirty, _ = extract_latents_and_labels(deep_ensemble[0], test_loader_dirty_mnist)
latents_test_fashion, _ = extract_latents_and_labels(deep_ensemble[0], test_loader_fashion)

# Compute Mahalanobis distance for uncertainties
uncertainties_dirty = compute_mahalanobis(latents_test_dirty, class_gaussians)
uncertainties_fashion = compute_mahalanobis(latents_test_fashion, class_gaussians)

# Combine uncertainties and labels
all_uncertainties = np.concatenate([uncertainties_dirty, uncertainties_fashion])
labels = np.concatenate([np.zeros_like(uncertainties_dirty), np.ones_like(uncertainties_fashion)])

# Plot ROC for Mahalanobis
plt.figure(figsize=(8, 6))
plot_roc_curve(torch.tensor(all_uncertainties), torch.tensor(labels), "Mahalanobis Distance")
plot_roc_curve(bald_scores, torch.tensor(labels), "BALD")
plot_roc_curve(pred_ent_scores, torch.tensor(labels), "Predictive Entropy")
plot_roc_curve(cond_ent_scores, torch.tensor(labels), "Conditional Entropy")
plt.show()
