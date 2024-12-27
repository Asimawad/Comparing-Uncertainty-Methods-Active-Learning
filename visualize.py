# Visualization tools for results

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional

def compute_accuracy_metric(predictive_log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for given predictions and labels."""
    predictions = predictive_log_probs.argmax(dim=1)
    return (predictions == labels).float().mean().item()

def compute_nll_metric(predictive_log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute negative log likelihood for given predictions and labels."""
    return -predictive_log_probs[range(len(labels)), labels].mean().item()

def plot_rejection_curve(
    uncertainties: torch.Tensor,
    predictive_log_probs: torch.Tensor,
    labels: torch.Tensor,
    method_name: str,
    performance_metric: Callable[[torch.Tensor, torch.Tensor], float] = compute_accuracy_metric,
    metric_name: str = "Accuracy",
    num_bins: int = 20,
    ax: Optional[plt.Axes] = None,
    device: str = "cuda"
) -> None:
    """Plot rejection curves for a given uncertainty metric.

    Args:
        uncertainties: Tensor of uncertainties for each sample.
        predictive_log_probs: Log probabilities of predictions.
        labels: Ground truth labels.
        method_name: Name of the method for the plot legend.
        performance_metric: Function to compute the metric (Accuracy, NLL, etc.).
        metric_name: Name of the performance metric for the y-axis.
        num_bins: Number of rejection rate bins.
        ax: Optional matplotlib Axes object for the plot.
        device: Device for computation.
    """
    sorted_indices = torch.argsort(uncertainties).to(device)
    sorted_predictive_log_probs = predictive_log_probs.to(device)[sorted_indices]
    sorted_labels = labels.to(device)[sorted_indices]
    total_samples = uncertainties.shape[0]
    rejection_rates = np.linspace(0, 1, num_bins)
    metric_values = []

    for rate in rejection_rates:
        num_kept = int(total_samples * (1 - rate))
        if num_kept > 0:
            metric_value = performance_metric(
                sorted_predictive_log_probs[:num_kept],
                sorted_labels[:num_kept]
            )
        else:
            metric_value = 1.0 if metric_name == "Accuracy" else 0.0
        metric_values.append(metric_value)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rejection_rates * 100, metric_values, label=method_name, marker='o', markersize=4)
    ax.set_xlabel('Rejection Rate (%)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs Rejection Rate')
    ax.grid(True, alpha=0.3)
    ax.legend()

if __name__ == "__main__":
    # Load saved methods
    methods = torch.load("./results/methods.pt")
    targets = torch.load("./results/targets.pt")  # Load saved targets (ensure it's saved during evaluation)

    # Plot Accuracy vs Rejection Rate
    fig, ax = plt.subplots(figsize=(10, 8))
    for method in methods:
        plot_rejection_curve(
            uncertainties=method["total_uncertainty"],
            predictive_log_probs=method["pred_dist"],
            labels=targets,
            method_name=method["name"],  # Add the method's name to the legend
            metric_name="Accuracy",
            num_bins=20,
            ax=ax  # Use the same axes for all methods
        )
    plt.show()

    # Plot NLL vs Rejection Rate
    fig, ax = plt.subplots(figsize=(10, 8))
    for method in methods:
        plot_rejection_curve(
            uncertainties=method["total_uncertainty"],
            predictive_log_probs=method["pred_dist"],
            labels=targets,
            method_name=method["name"],
            performance_metric=compute_nll_metric,
            metric_name="NLL",
            num_bins=20,
            ax=ax  # Use the same axes for all methods
        )
    plt.show()
