X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

pytorch_losses = []

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)

        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        # use sigmoid for binary classification
        x = torch.sigmoid(self.fc2(x))
        return x

model = XORNet()
# BCE because it will penaltize based on binary diffrence between 0-1
criterion = nn.BCELoss()
# Stochastic Gradient Descent will just take steps based on loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

def plot_decision_boundary(model, epoch):
    xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, 200),
                         np.linspace(-0.2, 1.2, 200))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    preds = model(grid).detach().numpy()
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y[:,0], edgecolors="k", cmap="RdBu")
    plt.title(f"Epoch {epoch}")
    plt.pause(0.1)
    plt.clf()

def get_weights(model):
    return {
        "fc1_weight": model.fc1.weight.data.clone(),
        "fc1_bias": model.fc1.bias.data.clone(),
        "fc2_weight": model.fc2.weight.data.clone(),
        "fc2_bias": model.fc2.bias.data.clone()
    }


plt.figure(figsize=(6,6))

for epoch in range(500):  # keep small for readability
    # FORWARD
    y_pred = model(X)
    loss = criterion(y_pred, y)
    pytorch_losses.append(loss.item())
    print(f"\nEpoch {epoch}")
    print(f"Loss BEFORE update: {loss.item():.6f}")

    # WEIGHTS BEFORE
    weights_before = get_weights(model)

    # BACKWARD
    optimizer.zero_grad()
    loss.backward()

    # UPDATE
    optimizer.step()

    # WEIGHTS AFTER
    weights_after = get_weights(model)

    # WEIGHT CHANGES
    print("Weight changes:")

    for key in weights_before:
        delta = weights_after[key] - weights_before[key]
        print(f"{key} mean Δ = {delta.mean().item():.6f}")
# VISUALIZE
plot_decision_boundary(model, epoch)
plt.show()
