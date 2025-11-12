# TensorFlow MLP: SGD vs Adam (nonlinear regression)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# --- Synthetic nonlinear data ---
N = 300
X = np.linspace(-2.5, 2.5, N).reshape(-1, 1).astype(np.float32)
# Nonlinear target: combination of sine + linear term + noise
y = (np.sin(2.5 * X) + 0.4 * X).astype(np.float32) + 0.2 * np.random.randn(N, 1).astype(np.float32)

# --- Build MLP factory ---
def build_mlp():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Create one model to get initial weights
base_model = build_mlp()
# Run one forward pass to ensure weights are created
_ = base_model(X[:2])

initial_weights = base_model.get_weights()  # save initial weights for fair start

# --- Training helper ---
def train_with_optimizer(optimizer, initial_weights, epochs=300, batch_size=32):
    model = build_mlp()
    _ = model(X[:2])  # ensure weights created
    model.set_weights(initial_weights)  # set same initial weights
    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, history.history['loss']

# --- Optimizers ---
sgd_opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
adam_opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# --- Train both ---
sgd_model, sgd_losses = train_with_optimizer(sgd_opt, initial_weights, epochs=400)
adam_model, adam_losses = train_with_optimizer(adam_opt, initial_weights, epochs=400)

# --- Plot training loss ---
plt.figure(figsize=(9,5))
plt.plot(sgd_losses, label='SGD + momentum (lr=0.01, momentum=0.9)', linewidth=2)
plt.plot(adam_losses, label='Adam (lr=0.01)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss: SGD vs Adam (MLP, nonlinear regression)')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot model fits vs data ---
X_test = np.linspace(-2.5, 2.5, 500).reshape(-1,1).astype(np.float32)
y_true = np.sin(2.5 * X_test) + 0.4 * X_test

y_sgd = sgd_model.predict(X_test)
y_adam = adam_model.predict(X_test)

plt.figure(figsize=(9,5))
plt.scatter(X, y, s=12, alpha=0.5, label='Training data (noisy)')
plt.plot(X_test, y_true, linestyle='--', linewidth=2, label='True function')
plt.plot(X_test, y_sgd, linewidth=2, label='SGD model prediction')
plt.plot(X_test, y_adam, linewidth=2, label='Adam model prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model fits: SGD vs Adam')
plt.legend()
plt.grid(True)
plt.show()

# --- Report final losses (last epoch) ---
print(f"Final training MSE (SGD):  {sgd_losses[-1]:.6f}")
print(f"Final training MSE (Adam): {adam_losses[-1]:.6f}")
