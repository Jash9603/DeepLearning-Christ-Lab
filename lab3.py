

# Setup

# Install and import required libraries
!pip install torch torchvision matplotlib scikit-learn


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 2. Load and Preprocess Dataset (Breast Cancer Dataset)"""

# Load dataset from sklearn
data = load_breast_cancer()
X = data.data
y = data.target


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)


X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

### Dataset Split

- 455 samples are used for training the model.
- 114 samples are used for testing the model.
- Most data is used for learning and a smaller part is used for evaluation.

# 3. Define Neural Network Model


class NeuralNet(nn.Module):
  def __init__(self, input_size):
    super(NeuralNet, self).__init__()
    self.model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
    )
  def forward(self, x):
    return self.model(x)

### Neural Network Architecture


# 4. Utility Functions


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()

            preds = (outputs >= 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, acc


def get_l1_penalty(model):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return l1_norm


def get_weight_statistics(model):
    weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            weights.append(param.detach().cpu().numpy().flatten())

    weights = np.concatenate(weights)

    stats = {
        "mean": np.mean(weights),
        "std": np.std(weights),
        "min": np.min(weights),
        "max": np.max(weights),
        "num_near_zero": np.sum(np.abs(weights) < 1e-3)
    }

    return weights, stats

### Utility Functions

# 5. Training Function (Supports L1, L2, ElasticNet)


def train_model(
    regularization="none",
    l1_lambda=0.0,
    l2_lambda=0.0,
    epochs=50
):

    model = NeuralNet(X_train.shape[1])
    criterion = nn.BCELoss()

    # L2 is handled by weight_decay in optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=l2_lambda
    )

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    weight_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            # Add L1 penalty if needed
            if regularization in ["l1", "elastic"]:
                l1_penalty = l1_lambda * get_l1_penalty(model)
                loss = loss + l1_penalty

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluation
        train_loss, train_acc = evaluate(model, train_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Store weights
        weights, _ = get_weight_statistics(model)
        weight_history.append(weights)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "weight_history": weight_history
    }


# 6. Train Models with Different Regularizations


print("\nTraining WITHOUT Regularization")
none_results = train_model(
    regularization="none",
    epochs=60
)

print("\nTraining with L2 Regularization")
l2_results = train_model(
    regularization="l2",
    l2_lambda=0.01,
    epochs=60
)

print("\nTraining with L1 Regularization")
l1_results = train_model(
    regularization="l1",
    l1_lambda=0.0005,
    epochs=60
)

print("\nTraining with Elastic Net")
elastic_results = train_model(
    regularization="elastic",
    l1_lambda=0.0005,
    l2_lambda=0.01,
    epochs=60
)


# 7. Plot Training & Testing Loss


plt.figure(figsize=(12, 6))

plt.plot(none_results["test_losses"], label="No Reg")
plt.plot(l2_results["test_losses"], label="L2")
plt.plot(l1_results["test_losses"], label="L1")
plt.plot(elastic_results["test_losses"], label="Elastic Net")

plt.xlabel("Epochs")
plt.ylabel("Test Loss")
plt.title("Test Loss Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 8. Plot Accuracy Comparison

plt.figure(figsize=(12, 6))

plt.plot(none_results["test_accs"], label="No Reg")
plt.plot(l2_results["test_accs"], label="L2")
plt.plot(l1_results["test_accs"], label="L1")
plt.plot(elastic_results["test_accs"], label="Elastic Net")

plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 9. Weight Distribution Analysis


def plot_weight_distribution(results, title):
    final_weights = results["weight_history"][-1]

    plt.figure(figsize=(8, 5))
    plt.hist(final_weights, bins=50)
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


plot_weight_distribution(none_results, "Weights: No Regularization")
plot_weight_distribution(l2_results, "Weights: L2 Regularization")
plot_weight_distribution(l1_results, "Weights: L1 Regularization")
plot_weight_distribution(elastic_results, "Weights: Elastic Net")

# 10. Weight Sparsity Analysis

def print_weight_stats(results, name):
    model = results["model"]
    weights, stats = get_weight_statistics(model)

    print(f"\n{name} Weight Statistics")
    print("---------------------------")
    print("Mean           :", round(stats["mean"], 6))
    print("Std Dev        :", round(stats["std"], 6))
    print("Min            :", round(stats["min"], 6))
    print("Max            :", round(stats["max"], 6))
    print("Near-Zero Count:", stats["num_near_zero"])
    print("Total Weights  :", len(weights))


print_weight_stats(none_results, "No Regularization")
print_weight_stats(l2_results, "L2 Regularization")
print_weight_stats(l1_results, "L1 Regularization")
print_weight_stats(elastic_results, "Elastic Net")

# 11. Final Performance Summary


def print_final_results(name, results):
    print(f"\n{name} Final Results")
    print("----------------------")
    print("Final Train Accuracy:", round(results["train_accs"][-1], 4))
    print("Final Test Accuracy :", round(results["test_accs"][-1], 4))
    print("Final Test Loss     :", round(results["test_losses"][-1], 4))


print_final_results("No Regularization", none_results)
print_final_results("L2 Regularization", l2_results)
print_final_results("L1 Regularization", l1_results)
print_final_results("Elastic Net", elastic_results)
