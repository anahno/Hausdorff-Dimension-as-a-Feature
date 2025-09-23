# Hausdorff-Dimension-as-a-Feature

Hausdorff Dimension as a Learnable Feature for Face Recognition
Authors: Behzad Farhadi, Behnam Farhadi

This repository contains the official Python implementation for the experiments described in our paper, "A Scalable Framework for Hausdorff-Weighted Persistence and Its Application as a Geometric Biomarker".

Abstract
This project introduces a novel feature engineering technique based on the estimated Hausdorff dimension (d_hat). The core hypothesis is that d_hat, which quantifies the intrinsic complexity of a class of data points, can serve as a powerful feature to improve the accuracy of machine learning classifiers. We validate this hypothesis on the Labeled Faces in the Wild (LFW) dataset, a standard benchmark for face recognition.

Our key finding is that by augmenting a strong PCA-based feature set with our single, geometrically-motivated d_hat feature, we achieved a significant 7.76% improvement in classification accuracy, increasing it from 82.61% to 90.37%.

Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
The code is written in Python 3. You will need the following libraries installed. You can install them using pip:

pip install numpy matplotlib scikit-learn

Installation
Clone the repository to your local machine:

git clone [https://github.com/anahno/Hausdorff-Dimension-as-a-Feature.git](https://github.com/anahno/Hausdorff-Dimension-as-a-Feature.git)

Navigate to the project directory:

cd Hausdorff-Dimension-as-a-Feature

Running the Experiment
The main script faces_analysis.py contains the complete pipeline for running the final experiment on the LFW dataset. To run the analysis, simply execute the script from your terminal:

python faces_analysis.py

The script will automatically download the LFW dataset (this may take a few minutes on the first run), perform the feature engineering steps, train the classifiers, and print the final results summary to the console.

Code Description
faces_analysis.py: The main executable script. It contains all functions for data loading, dimension estimation, and the classification experiment.

estimate_dimension_simple(): The core function that implements our robust Hausdorff dimension estimator.

run_classification_experiment(): The function that orchestrates the comparison between the baseline model (PCA only) and our augmented model (PCA + d_hat).

Citation
If you find this work useful in your research, please consider citing our paper:


License
This project is licensed under the MIT License - see the LICENSE.md file for details.
