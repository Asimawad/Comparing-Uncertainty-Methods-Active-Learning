# Training script for uncertainty methods

import torch
import torch.nn as nn
from models import BayesianMNISTCNN, create_model_optimizer
from data_utils import get_data_loaders
import os

#  Training loop
def train_model(model , optimizer , epochs , train_loader, device):
  criterion = nn.NLLLoss()
  model.to(device)
  model.train()  # Set model to training mode
  for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    for index ,(inputs, targets) in enumerate(train_loader):
      inputs , targets = inputs.to(device) , targets.to(device)

      # set gradients to zero
      optimizer.zero_grad()

      # calculate the forward pass -> predictions
      outputs, embeddings = model(inputs)

      # now loss calculations
      loss = criterion(outputs, targets)

      # now backprob
      loss.backward()
      # now parameter update
      optimizer.step()

      # this is to accomulate the loss over the batches, then calculate the average per batch by deviding by the no. of batches
      total_loss += loss.item()
      values,indices = outputs.max(1)
      total += targets.size(0)
      correct += indices.eq(targets).sum().item()

      # Print statistics every few batches
    print(f'Epoch [{epoch}], '
          f'Loss: {total_loss/(index+1):.4f}, '
          f'Accuracy: {100.*correct/total:.2f}%')
  model.to("cpu")
  return model


if __name__ == '__main__':
    train_loader, test_loader_mnist, test_loader_dirty_mnist, test_loader_fashion = get_data_loaders(batch_size=64, eval_batch_size=1024, device="cuda")

    # Directory to save trained models
    save_dir = "./deep_ensemble_trained"
    os.makedirs(save_dir, exist_ok=True)

    epochs = 10
    no_models = 8
    deep_ensemble = []

    models = [create_model_optimizer(i) for i in range(no_models)]
    for i, (model, optimizer) in enumerate(models):
        deep_ensemble.append(train_model(model, optimizer, epochs, train_loader))

    # Save each model in the ensemble
    for i, model in enumerate(deep_ensemble):
        save_path = os.path.join(save_dir, f"trained_model_{i}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Model {i} saved at {save_path}")
