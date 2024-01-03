# Fashion MNIST Solver

## Introduction

The Fashion MNIST Solver is a project aimed at solving the Fashion MNIST problem. Fashion MNIST is a dataset containing 60,000 grayscale images of 10 different fashion items, with 6,000 images per class. The goal of this project is to develop a machine learning model that can classify these fashion items with high accuracy.

## Dataset

The Fashion MNIST dataset consists of 60,000 training images and 10,000 testing images. Each image is a 28x28 pixel grayscale image representing a fashion item. The dataset includes the following classes:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model

The project uses various machine learning and deep learning techniques to build and train models for fashion item classification. The following approaches will be explored:

1. **Convolutional Neural Network (CNN)**: A simple CNN architecture will be used to extract meaningful features from the images and make predictions.

2. **Transfer Learning**: Pre-trained models such as ConvNext and ResNet50 will be fine-tuned for the Fashion MNIST dataset to leverage their feature extraction capabilities.

3. **Ensemble Methods**: Multiple models will be trained and combined to improve classification accuracy.

4. **Data augmentation**: Several data augmentation approaches will be explored to improve the performance of the classifier, incluiding the generation of synthetic data trough Gaussian Adversarial Networks.

## Evaluation Metrics

The performance of the models will be evaluated using the following metrics:

- **Accuracy**: The percentage of correctly classified images. This is suitable since the testing data set is balanced.

- **Confusion Matrix**: To understand the distribution of true positive, true negative, false positive, and false negative predictions for each class.


## Implementation

The implementation of the project will involve the following steps:

1. Data Preprocessing: Loading and preprocessing the Fashion MNIST dataset, including data augmentation.

2. Model Building: Creating and training machine learning and deep learning models using frameworks like TensorFlow and PyTorch.

3. Hyperparameter Tuning: Optimizing model hyperparameters to improve performance.

4. Evaluation: Evaluating the models using the evaluation metrics mentioned above. Compare individual predictions with ensembled predictions.

5. Visualization: Visualizing the results, including confusion matrices and learning curves.

## Code Organization

The project code will be organized into the following directories:

- `GAN data augmentation`: Contains a notebook for generating synthetic data used for data augmentation.

- `Models`: Contains model architectures, training notebooks, and pretrained models.


## Contributors

- [Omar Muñoz]
- [Leonardo Garrafa]
- [Israel Cárdenas]

## Disclaimer

The Fashion MNIST project is provided "as is" and "with all faults." We make no representations or warranties of any kind concerning the performance, accuracy, or suitability of the models and code provided in this project. Users are solely responsible for any risks associated with using or modifying the code and models. We will not be liable for any damages or inaccuracies that may arise from using this project.
