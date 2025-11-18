from google.colab import files

uploaded = files.upload()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load data
with open("classification_knn.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys() if hasattr(data, "keys") else None)
print(data[:5] if isinstance(data, (list, tuple)) else None)

# Extract data

X_train = data['x_train']
Y_train = data['y_train']
X_test = data['x_test']
Y_test = data['y_test']

print("Shapes:")
print("X_train", X_train.shape, "y_train:", Y_train.shape)
print("X_test", X_test.shape, "y_test:", Y_test.shape)

# Euclidean distance

def euclidean_distance(u, v):
  return np.sum((u - v) ** 2)

# Kernelized distance

def kernel(u, v):
  return (np.dot(u, u)) * (np.dot(v,v))

def kernelized_distance(u, v):
  k_uu = kernel(u, u)
  k_uv = kernel(u, v)
  k_vv = kernel(v, v)

  return k_uu - 2 * k_uv + k_vv


def knn_predict(X_train, Y_train, X_test, K, distance_fn):
  # Predicts labels for X_test using KNN with distnace function

  y_pred = []
  for x in X_test:
    distances = [distance_fn(x, x_train) for x_train in X_train]

    nn_indices = np.argsort(distances)[:K]

    nn_labels = [Y_train[i] for i in nn_indices]

    pred_label = Counter(nn_labels).most_common(1)[0][0]
    y_pred.append(pred_label)

  return np.array(y_pred)
def accuracy(y_true, y_pred):
  return np.mean(y_true == y_pred)

K_values = range(1,26)
acc_euclid = []
acc_kernel = []

for K in K_values:
  y_pred_euclid = knn_predict(X_train, Y_train, X_test, K, euclidean_distance)
  y_pred_kernel = knn_predict(X_train, Y_train, X_test, K, kernelized_distance)

  acc_euclid.append(accuracy(Y_test, y_pred_euclid))
  acc_kernel.append(accuracy(Y_test, y_pred_kernel))


# Plot Results

plt.figure(figsize=(8,5))
plt.plot(K_values, acc_euclid, 'o-', label = 'Euclidean Distance')
plt.plot(K_values, acc_kernel, 's--', label = 'Kernelized Distance')
plt.xlabel('K')
plt.ylabel('Test Accuracy')
plt.title('K-NN Accuracy vs K')
plt.legend()
plt.grid(True)
plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load data
with open("regression_kernels.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys() if hasattr(data, "keys") else None)
print(data[:5] if isinstance(data, (list, tuple)) else None)

X_train = data['x_train']
Y_train = data['y_train']
X_test = data['x_test']
Y_test = data['y_test']

print("Shapes:")
print("X_train", X_train.shape, "y_train:", Y_train.shape)
print("X_test", X_test.shape, "y_test:", Y_test.shape)

def mean_squared_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# Kernel functions

def poly_kernel(u, v, degree):
    return (1 + u @ v.T) ** degree

def rbf_kernel(u, v, gamma):
  u_norm = np.sum(u*u, axis=1)[:, None]
  v_norm = np.sum(v*v, axis=1)[None, :]
  d2 = u_norm + v_norm - 2 * u @ v.T
  return np.exp(-gamma * d2)

def custom_kernel(u, v, gamma):
  u_norm = np.sum(u*u, axis=1)[:, None]
  v_norm = np.sum(v*v, axis=1)[None, :]
  d2 = u_norm + v_norm - 2 * u @ v.T
  e = np.exp(-gamma * d2)
  return - e / (1 + e)

def kernel_regressor(X_train, Y_train, X_test, kernel_fn, **params):
  K = kernel_fn(X_test, X_train, **params)
  weights = K / np.sum(K, axis=1, keepdims=True)
  return weights @ Y_train


# Run experiments

degrees = [1, 2, 3, 4, 6]
gammas = [1, 10, 100, 1000]

results = []

# Polynomial
for d in degrees:
    pred = kernel_regressor(X_train, Y_train, X_test, poly_kernel, degree=d)
    mse = mean_squared_error(Y_test, pred)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
    ax[0].set_title('Ground truth')
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=pred)
    ax[1].set_title('Prediction')
    fig.suptitle(r'Kernel: Polynomial with degree=%d, MSE=%f' % (d, mse))
    plt.show()
    print(f"Polynomial kernel (degree={d}) MSE = {mse:.6f}")


# RBF kernel results

for gamma in gammas:
    pred = kernel_regressor(X_train, Y_train, X_test, rbf_kernel, gamma=gamma)
    mse = mean_squared_error(Y_test, pred)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
    ax[0].set_title('Ground truth')
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=pred)
    ax[1].set_title('Prediction')
    fig.suptitle(r'Kernel: RBF with $\gamma=$%0.01f, MSE=%f' % (gamma, mse))
    plt.show()
    print(f"RBF kernel (gamma={gamma}) MSE = {mse:.6f}")


# Custom kernel results

for gamma in gammas:
    pred = kernel_regressor(X_train, Y_train, X_test, custom_kernel, gamma=gamma)
    mse = mean_squared_error(Y_test, pred)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
    ax[0].set_title('Ground truth')
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=pred)
    ax[1].set_title('Prediction')
    fig.suptitle(r'Kernel: Custom with $\gamma=$%0.01f, MSE=%f' % (gamma, mse))
    plt.show()
    print(f"Custom kernel (gamma={gamma}) MSE = {mse:.6f}")

import time
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors


best_kernel_fn = custom_kernel
best_kernel_name = "Custom"
best_params = {'gamma': 1000}


# LSH helper functions

def lsh_neighbors(X_train, X_query, L=5, n_neighbors=50):
    """
    Approximate neighbors via random Gaussian projections.
    Returns a list of index arrays for each query.
    """
    proj = GaussianRandomProjection(n_components=L, random_state=42)
    X_train_proj = proj.fit_transform(X_train)
    X_query_proj = proj.transform(X_query)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_train_proj)
    return nn.kneighbors(X_query_proj, return_distance=False)

def kernel_regressor_lsh(X_train, Y_train, X_test, kernel_fn, L=5, n_neighbors=50, **params):
    """
    Approximate non-parametric regression using LSH neighbors only.
    Vectorized per-subset computation.
    """
    neighbors = lsh_neighbors(X_train, X_test, L=L, n_neighbors=n_neighbors)
    preds = np.zeros(len(X_test))
    for i, idxs in enumerate(neighbors):
        X_subset = X_train[idxs]
        Y_subset = Y_train[idxs]
        K = kernel_fn(X_test[i:i+1], X_subset, **params)       # (1 × n_neighbors)
        weights = K / np.sum(K, axis=1, keepdims=True)
        preds[i] = weights @ Y_subset
    return preds


# Compare runtimes (10 runs)

L = 5
n_neighbors = 50
runs = 10
t_no_lsh = t_lsh = 0.0

for _ in range(runs):
    start = time.time()
    _ = kernel_regressor(X_train, Y_train, X_test, best_kernel_fn, **best_params)
    t_no_lsh += time.time() - start

    start = time.time()
    _ = kernel_regressor_lsh(X_train, Y_train, X_test, best_kernel_fn,
                             L=L, n_neighbors=n_neighbors, **best_params)
    t_lsh += time.time() - start

t_no_lsh /= runs
t_lsh /= runs
print(f"\nAverage inference time over {runs} runs:")
print(f"Without LSH : {t_no_lsh:.4f} s")
print(f"With LSH    : {t_lsh:.4f} s")


# Evaluate accuracy & visualize

pred_full = kernel_regressor(X_train, Y_train, X_test, best_kernel_fn, **best_params)
pred_lsh  = kernel_regressor_lsh(X_train, Y_train, X_test, best_kernel_fn,
                                 L=L, n_neighbors=n_neighbors, **best_params)

mse_full = mean_squared_error(Y_test, pred_full)
mse_lsh  = mean_squared_error(Y_test, pred_lsh)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
ax[0].set_title("Ground Truth")
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=pred_full)
ax[1].set_title(f"Full {best_kernel_name}\nMSE={mse_full:.4f}")
ax[2].scatter(X_test[:, 0], X_test[:, 1], c=pred_lsh)
ax[2].set_title(f"LSH {best_kernel_name}\nMSE={mse_lsh:.4f}")
fig.suptitle(f"{best_kernel_name} Regression with and without LSH (L={L})\n"
             f"Avg Inference Time: {t_no_lsh:.4f}s → {t_lsh:.4f}s")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load data
with open("pca_images.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print("Keys:", data.keys() if hasattr(data, "keys") else None)

X = data["x_train"]        # shape (100, 784)
X = X / 255.0
X = X - np.mean(X, axis=0)   # center data

# Compute covariance using the transpose trick (N < d)
S = (X @ X.T) / X.shape[1]   # smaller 100x100 matrix

# Eigen-decomposition
eigvals, eigvecs_small = np.linalg.eig(S)

# Convert eigenvectors to the d-dimensional space
eigvecs = X.T @ eigvecs_small
eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

# Sort by eigenvalue
idx = np.argsort(-eigvals.real)
eigvecs = eigvecs[:, idx]

# Visualize first 10 eigenvectors
for i in range(10):
    plt.imshow(eigvecs[:, i].real.reshape(28, 28), cmap="gray")
    plt.title(f"Eigenvector {i+1}")
    plt.axis("off")
    plt.show()

x0 = X[0]                      # pick any sample
ks = [1, 5, 10, 50, 100]
mse_list = []

for k in ks:
    Wk = eigvecs[:, :k]
    x_rec = Wk @ (Wk.T @ x0)
    mse = np.mean((x0 - x_rec)**2)
    mse_list.append(mse)
    plt.imshow(x_rec.reshape(28, 28), cmap="gray")
    plt.title(f"k={k}, MSE={mse:.4f}")
    plt.axis("off")
    plt.show()

print("Reconstruction MSEs:", dict(zip(ks, mse_list)))
