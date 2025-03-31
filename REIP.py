
# Datasets:
#   - Structured data is obtained from the Real Estate Price Prediction Data on Figshare:
#     https://doi.org/10.6084/m9.figshare.26517325.v1
#
#   - Image data: A collection of property images 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# 1. Load Structured Real Estate Data
# -----------------------------
# The CSV is expected to contain features such as "Area", "Floor", "Num_Bedrooms", "Num_Bathrooms",
# "Property_Age", "Condition", "Proximity", and a target column "Price".

df = pd.read_csv("real_estate_price_prediction.csv")
print("Data shape:", df.shape)
print(df.head())

features = ['Area', 'Floor', 'Num_Bedrooms', 'Num_Bathrooms', 'Property_Age', 'Proximity']
target = 'Price'
X_struct = df[features].values
y_price = df[target].values


median_price = np.median(y_price)
y_invest = (y_price > median_price).astype(int)

# Normalizing numeric features for regression & logistic regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_struct_norm = scaler.fit_transform(X_struct)

# -----------------------------
# 2. Linear Regression: Predicting Property Prices
# -----------------------------
def linear_regression_GD(X, y, alpha=0.001, num_iters=5000):
    m, n = X.shape
    X_b = np.hstack([np.ones((m, 1)), X]) 
    theta = np.random.randn(n + 1)
    for it in range(num_iters):
        predictions = X_b.dot(theta)
        error = predictions - y
        gradients = (2 / m) * X_b.T.dot(error)
        theta = theta - alpha * gradients
    return theta

theta_lin = linear_regression_GD(X_struct_norm, y_price)
print("Learned parameters (linear regression):", theta_lin)

# Predict prices
X_b = np.hstack([np.ones((X_struct_norm.shape[0], 1)), X_struct_norm])
price_pred = X_b.dot(theta_lin)

# Plot Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
plt.scatter(y_price, price_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted Property Prices")
plt.show()

# -----------------------------
# 3. Logistic Regression: Classify Investment Quality
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_GD(X, y, alpha=0.01, num_iters=5000):
    m, n = X.shape
    X_b = np.hstack([np.ones((m, 1)), X])
    theta = np.random.randn(n + 1)
    for it in range(num_iters):
        z = X_b.dot(theta)
        predictions = sigmoid(z)
        error = predictions - y
        gradients = (1 / m) * X_b.T.dot(error)
        theta = theta - alpha * gradients
    return theta

theta_log = logistic_regression_GD(X_struct_norm, y_invest)
print("Learned parameters (logistic regression):", theta_log)

# Predicting investment probability
z_log = X_b.dot(theta_log)
investment_proba = sigmoid(z_log)

plt.figure(figsize=(8, 5))
plt.scatter(np.arange(len(y_invest)), y_invest, color='red', label='True Label', alpha=0.6)
plt.scatter(np.arange(len(y_invest)), investment_proba, color='blue', label='Predicted Probability', alpha=0.6)
plt.xlabel("Property Index")
plt.ylabel("Probability / Label")
plt.title("Logistic Regression: Investment Probability vs True Label")
plt.legend()
plt.show()

# -----------------------------
# 4. Neural Network: Image-Based Investment Signal
# -----------------------------
# Image filenames are "0.jpg", "1.jpg", ..., "m-1.jpg"
# corresponding to the rows in our CSV file.

IMG_HEIGHT, IMG_WIDTH = 64, 64  # Resizing images to a fixed size
image_folder = "property_images"

def load_property_images(folder, num_images):
    images = []
    for i in range(num_images):
        img_path = os.path.join(folder, f"{i}.jpg")
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
            img_array = img_to_array(img) / 255.0  # Normalizing pixel values
        else:
            img_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))
        images.append(img_array)
    return np.array(images)

# Load images for all properties (ensure num_images equals number of rows in df)
X_images = load_property_images(image_folder, df.shape[0])
print("Images shape:", X_images.shape)

# CNN model for image-based classification
model_img = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_img.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model_img.summary()

# Training the image model using the same investment quality labels from logistic regression.
history = model_img.fit(X_images, y_invest, epochs=10, batch_size=16, verbose=2)

# Predicting image-based investment probability
investment_proba_img = model_img.predict(X_images).flatten()

# -----------------------------
# 5. Fusion: Combining the Three Models
# -----------------------------
price_pred_norm = (price_pred - price_pred.min()) / (price_pred.max() - price_pred.min())

w1, w2, w3 = 1/3, 1/3, 1/3
final_score = w1 * price_pred_norm + w2 * investment_proba + w3 * investment_proba_img
final_classification = (final_score > 0.5).astype(int)

# Plotting the final scores against true investment labels
plt.figure(figsize=(8, 5))
plt.scatter(np.arange(len(y_invest)), y_invest, color='red', marker='o', label='True Investment Label', alpha=0.6)
plt.scatter(np.arange(len(y_invest)), final_score, color='blue', marker='x', label='Final Investment Score', alpha=0.6)
plt.xlabel("Property Index")
plt.ylabel("Score / Label")
plt.title("Final Investment Score vs True Investment Label")
plt.legend()
plt.show()

# Printing sample final decisions
for i in range(5):
    print(f"Property {i}:")
    print(f"  Predicted Price: {price_pred[i]:.2f} (Normalized: {price_pred_norm[i]:.2f})")
    print(f"  Logistic Prob.: {investment_proba[i]:.2f}")
    print(f"  Image Model Prob.: {investment_proba_img[i]:.2f}")
    print(f"  Final Score  : {final_score[i]:.2f} -> Classified as: {final_classification[i]}")
    print("-----------------------------------------------------")

# =============================================================================
# Conclusion:
# This project integrates real-world data from a real estate price prediction dataset
# and property images to build a comprehensive investment analysis system.
# =============================================================================
