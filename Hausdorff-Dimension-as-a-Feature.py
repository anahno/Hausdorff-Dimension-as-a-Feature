# -*- coding: utf-8 -*-
"""
This script performs a feature engineering experiment on the Labeled Faces in the Wild (LFW) dataset.
The core hypothesis is that the estimated Hausdorff dimension (d_hat) of an individual's face images,
which quantifies the intrinsic complexity of their appearance variations, can serve as a powerful
new feature to improve classification accuracy.

The experiment follows these steps:
1.  Loads the LFW dataset, focusing on individuals with at least 70 images.
2.  Establishes a strong baseline classifier using Principal Component Analysis (PCA) for feature extraction.
3.  Calculates the d_hat value for each person using a robust correlation dimension estimator.
4.  Creates an augmented feature set by combining the PCA features with the new d_hat feature.
5.  Trains and evaluates two SVM classifiers: one on the baseline PCA features and one on the
    augmented feature set.
6.  Compares the accuracies to demonstrate the predictive value added by the d_hat feature.

Authors: Behzad Farhadi, Behnam Farhadi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. DATASET GENERATION
# ==============================================================================

def generate_faces_data():
    """
    Loads and preprocesses the Labeled Faces in the Wild (LFW) dataset.

    This function downloads the LFW dataset (if not already cached), selects a subset
    of individuals with a sufficient number of images for robust analysis, and applies
    standard scaling to the pixel data.

    Returns:
        X_scaled (np.ndarray): The scaled, high-dimensional image data.
        y (np.ndarray): The integer labels corresponding to each person.
    """
    # We select people with at least 70 images to ensure each class has enough
    # samples to estimate its intrinsic dimension.
    # resize=0.4 makes images smaller, which speeds up all subsequent computations.
    print("Loading Labeled Faces in the Wild (LFW) dataset...")
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    
    # StandardScaler is crucial for distance-based algorithms like PCA and SVM.
    # It centers the data and scales it to unit variance.
    X_scaled = StandardScaler().fit_transform(X)
    print(f"Loaded LFW dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Number of unique people: {len(np.unique(y))}")
    return X_scaled, y

# ==============================================================================
# 2. ALGORITHM: HAUSDORFF DIMENSION ESTIMATOR
# ==============================================================================

def estimate_dimension_simple(P, class_name=""):
    """
    A robust Hausdorff dimension estimator based on the correlation dimension.

    This algorithm estimates the intrinsic dimension of a point cloud by analyzing
    the power-law relationship between scales and the number of point pairs
    within those scales. The dimension is the slope of the log-log plot.

    Args:
        P (np.ndarray): The input point cloud for a single class.
        class_name (str): The name of the class (for printing purposes).

    Returns:
        float: The estimated Hausdorff dimension (d_hat).
    """
    n_points = P.shape[0]
    if n_points < 10: return 1.0

    # For small classes, use all points to maximize information.
    sample = P
    
    # Calculate all pairwise distances within the sample.
    distances = np.linalg.norm(sample[:, np.newaxis, :] - sample[np.newaxis, :, :], axis=2)
    # Filter out zero distances (a point to itself).
    unique_dists = np.unique(distances[distances > 1e-9])
    
    if len(unique_dists) < 5: return 1.0 # Fallback if not enough distinct distances.
    
    # Select a range of scales based on data distribution using percentiles.
    # This is a data-driven heuristic that makes the method robust.
    scales = np.percentile(unique_dists, np.linspace(20, 80, 10))
    # Count how many pairs of points are within each scale.
    counts = np.array([np.sum(distances < s) for s in scales])
    
    # Ensure we have valid data for log-log regression.
    valid_mask = (counts > 0) & (scales > 0)
    if np.sum(valid_mask) < 2: return 1.0

    log_scales = np.log(scales[valid_mask])
    log_counts = np.log(counts[valid_mask])
    
    try:
        # The slope of the line in the log-log plot is the correlation dimension.
        coeffs = np.polyfit(log_scales, log_counts, 1)
        return max(1.0, coeffs[0]) # Dimension cannot be less than 1.
    except np.linalg.LinAlgError:
        return 1.0 # Fallback in case of numerical instability.

# ==============================================================================
# 3. MACHINE LEARNING CLASSIFICATION
# ==============================================================================

def run_classification_experiment(X_full, y_full):
    """
    Conducts the main classification experiment to evaluate the d_hat feature.

    This function compares two models:
    1. Baseline: An SVM classifier trained on standard PCA features.
    2. Our Method: An SVM classifier trained on PCA features augmented with our d_hat feature.

    Args:
        X_full (np.ndarray): The complete, scaled dataset.
        y_full (np.ndarray): The labels for the complete dataset.
    """
    
    # --- Step 1: Baseline PCA Feature Engineering ---
    print("\n" + "="*60)
    print("Step 1: Creating baseline features using PCA")
    print("="*60)
    
    # 150 components is a standard choice for LFW, capturing significant variance.
    # 'randomized' SVD is faster for large matrices.
    n_pca_components = 150
    pca = PCA(n_components=n_pca_components, random_state=42, svd_solver='randomized')
    X_pca = pca.fit_transform(X_full)
    print(f"Reduced all {X_full.shape[0]} images to {n_pca_components} PCA components.")
    
    # --- Step 2: Our Feature Engineering (PCA + d_hat) ---
    print("\n" + "="*60)
    print("Step 2: Creating augmented features with d_hat")
    print("="*60)

    unique_labels = np.unique(y_full)
    num_classes = len(unique_labels)
    # A map is needed because LFW labels might not be contiguous (e.g., 0, 1, 3...).
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Calculate d_hat for each person (class) once.
    d_hat_per_class = np.zeros(num_classes)
    for original_label, new_idx in label_map.items():
        class_mask = (y_full == original_label)
        X_class = X_full[class_mask]
        d_hat_per_class[new_idx] = estimate_dimension_simple(X_class)

    # Create a feature vector where each image is assigned the d_hat of its person.
    d_hat_feature_vector = np.array([d_hat_per_class[label_map[label]] for label in y_full]).reshape(-1, 1)

    # Augment the PCA features with our new d_hat feature.
    X_augmented = np.hstack([X_pca, d_hat_feature_vector])
    print(f"Created augmented feature set with {X_augmented.shape[1]} features.")

    # --- Step 3: Train and Evaluate Baseline Model (PCA only) ---
    print("\n" + "="*60)
    print("Step 3: Training and Evaluating Baseline Classifier (PCA only)")
    print("="*60)
    
    # Scale the PCA features to have zero mean and unit variance.
    scaler_pca = StandardScaler()
    X_pca_scaled = scaler_pca.fit_transform(X_pca)
    
    # Split data into training and test sets. `stratify` ensures that the proportion
    # of images for each person is the same in both sets, which is crucial for fair evaluation.
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(
        X_pca_scaled, y_full, test_size=0.25, random_state=42, stratify=y_full
    )
    
    # Use a powerful RBF kernel SVM. Hyperparameters are tuned for this dataset.
    svm_pca = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
    svm_pca.fit(X_train_pca, y_train)
    y_pred_pca = svm_pca.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    
    print(f"Model: RBF Support Vector Machine (SVM)")
    print(f"Features Used: {n_pca_components} Scaled Principal Components")
    print(f"Accuracy on test set: {accuracy_pca:.2%}")

    # --- Step 4: Train and Evaluate Our Model (PCA + d_hat) ---
    print("\n" + "="*60)
    print("Step 4: Training and Evaluating Our Classifier (PCA + d_hat)")
    print("="*60)
    
    # Scale the augmented feature set. It's important to scale the d_hat feature along with PCA features.
    scaler_aug = StandardScaler()
    X_augmented_scaled = scaler_aug.fit_transform(X_augmented)
    
    # Use the exact same train/test split for a fair comparison.
    X_train_aug, X_test_aug, _, _ = train_test_split(
        X_augmented_scaled, y_full, test_size=0.25, random_state=42, stratify=y_full
    )

    svm_augmented = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42)
    svm_augmented.fit(X_train_aug, y_train)
    y_pred_aug = svm_augmented.predict(X_test_aug)
    accuracy_augmented = accuracy_score(y_test, y_pred_aug)

    print(f"Model: RBF Support Vector Machine (SVM)")
    print(f"Features Used: {n_pca_components} Scaled PCA Components + Scaled d_hat feature")
    print(f"Accuracy on test set: {accuracy_augmented:.2%}")

    # --- Step 5: Final Results Summary ---
    print("\n" + "="*60)
    print("                           RESULTS SUMMARY")
    print("="*60)
    header = "| {:<45} | {:<15} |".format("Feature Set", "Classification Accuracy")
    print(header)
    print("-" * (len(header) + 1))
    row1 = "| {:<45} | {:<15.2%} |".format(f"Baseline: Scaled PCA ({n_pca_components} features)", accuracy_pca)
    print(row1)
    row2 = "| {:<45} | {:<15.2%} |".format(f"Our Method: Scaled PCA + d_hat ({n_pca_components + 1} feat.)", accuracy_augmented)
    print(row2)
    print("-" * (len(header) + 1))
    
    improvement = accuracy_augmented - accuracy_pca
    if improvement > 0.0001:
        print(f"\nConclusion: Adding the scaled d_hat feature improved accuracy by {improvement:.2%}.")
    elif improvement < -0.0001:
        print(f"\nConclusion: Adding the scaled d_hat feature decreased accuracy by {-improvement:.2%}.")
    else:
        print("\nConclusion: Adding the scaled d_hat feature did not significantly change the accuracy.")

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # This is the main entry point of the script.
    X_scaled, y = generate_faces_data()
    run_classification_experiment(X_scaled, y)
    print("\nAnalysis complete.")

