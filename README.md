# MODELING Z-BOSON DECAY EVENTS WITH GENERATIVE MODELS

## **Project Overview**

This project focuses on the application of deep generative models to simulate **Z boson decay events**. Z bosons, fundamental particles mediating the weak force, decay into either muon-antimuon or electron-positron pairs. The primary goal of this study is to explore how **Variational Autoencoders (VAEs)** and **Wasserstein Generative Adversarial Networks (WGANs)** can be used to efficiently model these decay events, providing a faster and more efficient alternative to traditional Monte Carlo simulations. We also train a **Multi-Layer Perceptron (MLP)** for classification tasks to distinguish between the two decay channels.
Tha project has been carried out for the course in **Artificial Intelligence**

## 📊 Key Objectives

- **Modeling Z boson decays** using generative models (VAEs and WGANs) to simulate the particle decay processes.
- **Classifying decay events** into two classes: Z→µ⁺µ⁻ and Z→e⁺e⁻ using supervised learning.
- **Exploring data augmentation** techniques to improve model performance, including Mixup, KNN, and WGAN.
- **Evaluating model performance** using standard metrics, such as accuracy, ROC curve, and confusion matrix.

## 📄 Methodology

1. **Data Preprocessing**
   - Removal of irrelevant features, logarithmic transformation of skewed variables, outlier removal, and standardization of the features.
   
2. **Exploratory Data Analysis**
   - Dimensionality reduction techniques like PCA, t-SNE, and UMAP to visualize the feature space.
   - Spectral and Hierarchical Clustering to uncover natural groupings in the data.

3. **Generative Modeling**
   - **Variational Autoencoders (VAEs)**: Applied to model the distribution of the decay events, with improvements like circular loss functions for angular features.
   - **Wasserstein GANs (WGANs)**: Used to generate more realistic synthetic data that adheres to the physical constraints of Z boson decays.

4. **Classification with MLP**
   - Trained a Multi-Layer Perceptron (MLP) for classification using various techniques, including hyperparameter optimization via RandomizedSearchCV.
   - Utilized data augmentation methods, including **Mixup**, **KNN**, **Gaussian Mixture Models**, **Forest-based synthesis**, **Rule-based synthesis**, and **WGAN-based augmentation**.

5. **Regularization and Performance Optimization**
   - Implemented **dropout** regularization using TensorFlow to further improve the MLP's performance.

6. **Evaluation**
   - Models evaluated based on training, validation, and test accuracy, with ROC curve analysis and confusion matrices to measure classification effectiveness.

## 🎯 Key Findings

- **VAE**: Provides interpretable latent representations but struggles with angular features, which were mitigated by custom loss functions.
- **WGAN**: Produced sharper and more realistic synthetic decay samples with low Wasserstein distance.
- **MLP Classifier**: Achieved an accuracy of **89.8%** on the validation set using dropout regularization, and **89.5%** on the test set.
- **Best Data Augmentation**: WGAN-based augmentation showed promising results but needed to be carefully balanced to avoid performance degradation with too much synthetic data.

## 🧑‍💻 Tools & Technologies

- **Python** (NumPy, SciPy, Matplotlib, Scikit-learn, TensorFlow)
- **Deep Generative Models**: VAEs, WGANs
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Machine Learning**: MLPClassifier, Hyperparameter tuning (RandomizedSearchCV)

## 📁 Dataset

The dataset used in this project contains Z boson decay events, with features representing various physical properties of the particles. The dataset is available at [Kaggle](https://www.kaggle.com/datasets/omidbaghchehsaraei/identification-of-two-modes-of-z-boson/data).

## 📌 Conclusion

This project demonstrates the viability of generative models like VAEs and WGANs for simulating particle physics decay events efficiently. The results also show that advanced deep learning techniques, such as MLP with dropout regularization and WGAN-based data augmentation, can be effectively employed to solve complex classification problems in high-energy physics.
