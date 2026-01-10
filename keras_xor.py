import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

X = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
], dtype=np.float32)

y = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
], dtype=np.float32)


keras_losses = []

model_keras = tf.keras.Sequential([
    tf.keras.layers.Dense(
        8,
        activation="tanh",
        input_shape=(2,),
        name="fc1"
    ),
    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="fc2"
    )
])

def plot_decision_boundary(model_keras, epoch):
    xx, yy = np.meshgrid(
        np.linspace(-0.2, 1.2, 200),
        np.linspace(-0.2, 1.2, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model_keras(grid, training=False).numpy()
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:,0], X[:,1], c=y[:,0], edgecolors="k", cmap="RdBu")
    plt.title(f"Epoch {epoch}")
    plt.pause(0.1)
    plt.clf()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

def get_named_params(model_keras):
    params = {}
    for layer in model_keras.layers:
        for weight in layer.weights:
            params[weight.name] = weight.numpy().copy()
    return params


def get_params(model_keras):
    params = {}
    for layer in model_keras.layers:
        for weight in layer.weights:
            params[weight.name] = weight.numpy().copy()
    return params


plt.figure(figsize=(6,6))

for epoch in range(500):  # keep small for clarity

    # FORWARD 
    with tf.GradientTape() as tape:
        y_pred = model_keras(X, training=True)
        loss = loss_fn(y, y_pred)
        keras_losses.append(loss)
    print(f"\nEpoch {epoch}")
    print(f"Loss BEFORE update: {loss.numpy():.6f}")

    # PARAMETERS BEFORE 
    params_before = get_params(model_keras)

    

    # BACKWARD 
    grads = tape.gradient(loss, model_keras.trainable_variables)

    print("Gradient magnitudes:")
    for var, grad in zip(model_keras.trainable_variables, grads):
        print(f"{var.name}: mean grad = {tf.reduce_mean(grad).numpy():.6f}")

    # UPDATE 
    optimizer.apply_gradients(zip(grads, model_keras.trainable_variables))

    # PARAMETERS AFTER 
    params_after = get_params(model_keras)

    # PARAMETER CHANGES 
    print("How much the model_keras changed this step:")
    for name in params_before:
        change = np.abs(params_after[name] - params_before[name]).mean()
        print(f"{name} changed by ~ {change:.6f}")

# VISUALIZE 
plot_decision_boundary(model_keras, epoch)
plt.show()
