# Real Estate Investment Analysis System


An end-to-end machine learning system that combines structured property data and visual features to predict real estate prices and evaluate investment potential through model fusion.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Techniques & Concepts](#techniques--concepts)
- [Acknowledgments](#acknowledgments)

## Features
- **Multi-Modal Analysis**: Combines structured data and property images
- **Three Core Models**:
  - Linear Regression with Gradient Descent (Price Prediction)
  - Regularized Logistic Regression (Investment Classification)
  - Convolutional Neural Network (Image Analysis)
- **Ensemble Learning**: Meta-model combining predictions
- **Production-Ready**: Persistent scalers and reproducible results
- **Comprehensive Validation**: Early stopping, train-test splits, and performance metrics

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

```bash
# Clone repository
git clone https://github.com/Abhinav0821/Real-Estate-Investment-Analysis.git
cd Real-Estate-Investment-Analysis

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The project utilizes two key datasets:
1. **Structured Data**: A CSV file (`real_estate_price_prediction.csv`) containing real estate features such as:
   - Area, Floor, Number of Bedrooms, Number of Bathrooms
   - Property Age, Proximity, Condition, and Price

2. **Image Data**: A collection of grayscale property images stored in the `property_images/` directory. Images are resized to 64x64 pixels.

## Usage

To train and evaluate the model:

```bash
python train.py
```

To make predictions on new data:

```bash
python predict.py --input new_data.csv
```

## Model Details

- **Linear Regression**: Uses gradient descent for price prediction.
- **Logistic Regression**: Regularized logistic regression for investment classification.
- **CNN**: A convolutional neural network analyzing property images.
- **Fusion Model**: Combines outputs from the three models into a final investment score.

## Results

- **Price Prediction**: Evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- **Investment Classification**: Evaluated using accuracy and precision-recall metrics.
- **Fusion Model Performance**: Combines predictions from structured data and image analysis to improve investment classification accuracy.

## Techniques & Concepts

- **Feature Engineering**: Interaction terms added to structured data
- **Normalization**: Standard scaling applied to numeric features
- **Loss Functions**: MSE, MAE, and Huber loss used for different objectives
- **Neural Networks**: Fully connected dense layers and CNN layers for image processing
- **Ensemble Learning**: Weighted combination of three model predictions


## Acknowledgments

- Dataset provided by [Figshare](https://doi.org/10.6084/m9.figshare.26517325.v1)
- TensorFlow and Keras for deep learning implementation

