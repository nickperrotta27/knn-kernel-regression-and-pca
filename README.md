# KNN, Kernel Regression, and PCA Analysis

This project implements three core machine learning techniques from scratch:

- **K-Nearest Neighbors Classification**
- **Kernel Regression** with Polynomial, RBF, and Custom Kernels
- **Principal Component Analysis (PCA)** with Eigenvector Visualization

The datasets are provided as `.pkl` files and include synthetic classification data, regression surfaces, and image data for dimensionality reduction.

## Project Structure

```
├── classification_knn.pkl      # Dataset for KNN classification
├── regression_kernels.pkl      # Dataset for kernel regression
├── pca_images.pkl              # Dataset for PCA / eigenfaces
├── model.py                    # Main analysis notebook
└── README.md
```

## 1. KNN Classification

Implements K-Nearest Neighbors from scratch:

- Euclidean distance
- Custom "kernelized distance"
- Majority-vote prediction
- Accuracy evaluation
- Accuracy curves for K = 1 to 25

### Visualization

Test accuracy vs. K for both distance metrics.

## 2. Kernel Regression

Implements three kernel functions:

- **Polynomial kernel**
- **RBF (Gaussian) kernel**
- **Custom kernel**:
  
  $$K(u,v) = -\frac{e^{-\gamma \|u-v\|^2}}{1+e^{-\gamma \|u-v\|^2}}$$

### Includes

- Kernel matrix computation
- Weighted regression
- 2D scatter visualizations of ground-truth vs. predicted surfaces
- MSE calculation
- Hyperparameter sweeps (degree, gamma)

### LSH-Accelerated Kernel Regressor

Also includes an LSH-accelerated kernel regressor:

- Random Gaussian projections
- Approximate nearest neighbors
- Runtime comparison
- Quality vs. speed visualization

## 3. PCA & Eigenvector Visualization

Performs PCA manually using the "transpose trick":

- Centering
- Covariance via $XX^{\top}$ (because $N < d$)
- Eigenvalue decomposition
- Transformation back to original space
- Visualization of first 10 eigenvectors (as 28×28 images)

These eigenvectors represent principal directions in the dataset (similar to eigenfaces).

## Technologies Used

- Python
- NumPy
- Matplotlib
Scikit-learn (for LSH baseline only)
