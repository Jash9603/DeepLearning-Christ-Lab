
"""# Advanced task 1: Experimenting with Network Depth and Width"""

# I will be experimenting by manipulating number of layers and neurons
# to do that, There will be a flexible FNN first
class FlexibleFeedForwardNN(nn.Module):
    def __init__(self, hidden_layers, hidden_units):
        """
        hidden_layers: number of hidden layers
        hidden_units: neurons per hidden layer
        """
        super().__init__()

        layers = []
        input_size = 784

        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            input_size = hidden_units

        layers.append(nn.Linear(hidden_units, 10))  # Output layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Training and evaluation function so that I can reuse it
def train_and_evaluate(model, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracies.append(correct / total)

    return train_losses, test_accuracies

# running experiments with different layers and neurons
configs = [
    (1, 128),
    (3, 256),
    (5, 256)
]

results = {}

for layers, units in configs:
    model = FlexibleFeedForwardNN(layers, units).to(device)
    losses, accs = train_and_evaluate(model, train_loader, test_loader)
    results[f"{layers}_layers"] = (losses, accs)

# visualization
plt.figure(figsize=(12, 5))

# Training loss comparison
plt.subplot(1, 2, 1)
for key, (losses, _) in results.items():
    plt.plot(losses, label=key)
plt.title("Training Loss vs Depth")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Test accuracy comparison
plt.subplot(1, 2, 2)
for key, (_, accs) in results.items():
    plt.plot(accs, label=key)
plt.title("Test Accuracy vs Depth")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

"""Interpretation
- 1 layer was faster but it underfitted
- 3 layers were a balanced approached
- 5 layers were slower but it performed well

# Advanced Task 2: Experimenting with different Activation Functions
"""

# just a reusable function to set the activation function
def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.01)

class ActivationNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        act = get_activation(activation)

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# training with different activation functions and storing the results
activations = ["relu", "sigmoid", "tanh", "leaky_relu"]
activation_results = {}

for act in activations:
    model = ActivationNN(act).to(device)
    losses, accs = train_and_evaluate(model, train_loader, test_loader)
    activation_results[act] = (losses, accs)

# Plotting the comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for act, (losses, _) in activation_results.items():
    plt.plot(losses, label=act)
plt.title("Training Loss vs Activation")
plt.legend()

plt.subplot(1, 2, 2)
for act, (_, accs) in activation_results.items():
    plt.plot(accs, label=act)
plt.title("Test Accuracy vs Activation")
plt.legend()

plt.tight_layout()
plt.show()

"""Interpretation
- Relu wins among all because it is faster (doesn't have any big calculations) and also it doesn't have the vanishing gradient problem.
- Sigmoid causes vanishing gradient problem, because as we move forward in a deep neural network, it will make the value of neurons' output smaller and smaller. At one point, we won't be able to extract any information.
"""

