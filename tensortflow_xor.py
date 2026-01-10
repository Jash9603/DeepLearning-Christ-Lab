
X = tf.constant([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], dtype=tf.float32)

y = tf.constant([
    [0.],
    [1.],
    [1.],
    [0.]
], dtype=tf.float32)

tf_low_losses = []

# fc1: 2 → 8
W1 = tf.Variable(tf.random.normal([2, 8]), name="fc1_weight")
b1 = tf.Variable(tf.zeros([8]), name="fc1_bias")

# fc2: 8 → 1
W2 = tf.Variable(tf.random.normal([8, 1]), name="fc2_weight")
b2 = tf.Variable(tf.zeros([1]), name="fc2_bias")

def forward(x):
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.tanh(z1)
    z2 = tf.matmul(a1, W2) + b2
    y_pred = tf.nn.sigmoid(z2)
    return y_pred


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-7  # numerical safety
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    return -tf.reduce_mean(
        y_true * tf.math.log(y_pred) +
        (1 - y_true) * tf.math.log(1 - y_pred)
    )


def plot_decision_boundary(epoch):
    xx, yy = np.meshgrid(
        np.linspace(-0.2, 1.2, 200),
        np.linspace(-0.2, 1.2, 200)
    )

    grid = tf.constant(np.c_[xx.ravel(), yy.ravel()], dtype=tf.float32)
    preds = forward(grid).numpy().reshape(xx.shape)

    plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y[:,0], edgecolors="k", cmap="RdBu")
    plt.title(f"Epoch {epoch}")
    plt.pause(0.1)
    plt.clf()

optimizer = tf.optimizers.SGD(learning_rate=0.5)

plt.figure(figsize=(6,6))

for epoch in range(500):

    with tf.GradientTape() as tape:
        y_pred = forward(X)
        loss = binary_cross_entropy(y, y_pred)
        tf_low_losses.append(loss)

    print(f"\nEpoch {epoch}")
    print(f"Loss BEFORE update: {loss.numpy():.6f}")

    W1_before = W1.numpy().copy()
    b1_before = b1.numpy().copy()
    W2_before = W2.numpy().copy()
    b2_before = b2.numpy().copy()

    grads = tape.gradient(loss, [W1, b1, W2, b2])
    optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2]))

    print("Parameter updates (mean of changes):")
    print(f"fc1_weight mean of changes = {(W1.numpy() - W1_before).mean():.6f}")
    print(f"fc1_bias   mean of changes  = {(b1.numpy() - b1_before).mean():.6f}")
    print(f"fc2_weight mean of changes = {(W2.numpy() - W2_before).mean():.6f}")
    print(f"fc2_bias   mean of changes  = {(b2.numpy() - b2_before).mean():.6f}")
plot_decision_boundary(epoch)
plt.show()


optimizer = tf.optimizers.SGD(learning_rate=0.5)
