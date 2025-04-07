
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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os

np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# -----------------------------
# 1. Enhanced Data Loading & Preprocessing
# -----------------------------
def load_data(test_size=0.2):
    # Load and validate structured data
    df = pd.read_csv("real_estate_price_prediction.csv")
    assert df.shape[0] > 0, "No data loaded"
    assert 'property_id' in df.columns, "Dataset must contain 'property_id' column"
    
    # Feature engineering
    features = ['Area', 'Floor', 'Num_Bedrooms', 'Num_Bathrooms', 'Property_Age', 'Proximity']
    target = 'Price'
    
    # Splitting data with preserved IDs
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    price_threshold = np.percentile(train_df[target], 75)
    y_train_invest = (train_df[target] > price_threshold).astype(int)
    y_test_invest = (test_df[target] > price_threshold).astype(int)
    
    return (X_train, X_test, 
            train_df[target].values, test_df[target].values,
            y_train_invest, y_test_invest,
            train_df['property_id'].values, test_df['property_id'].values)

# -----------------------------
# 2. Improved Linear Regression with Convergence
# -----------------------------
class LinearRegressionGD:
    def __init__(self, alpha=0.01, tol=1e-4, max_iters=10000):
        self.alpha = alpha
        self.tol = tol
        self.max_iters = max_iters
        
    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.theta = np.random.randn(X_b.shape[1])
        prev_loss = float('inf')
        
        for it in range(self.max_iters):
            predictions = X_b.dot(self.theta)
            error = predictions - y
            loss = np.mean(error**2)
            
            if abs(prev_loss - loss) < self.tol:
                break
                
            gradients = 2/len(X) * X_b.T.dot(error)
            self.theta -= self.alpha * gradients
            prev_loss = loss
            
        return self
    
    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b.dot(self.theta)

# -----------------------------
# 3. Regularized Logistic Regression
# -----------------------------
class LogisticRegressionGD:
    def __init__(self, alpha=0.1, lambda_=0.1, tol=1e-4, max_iters=5000):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iters = max_iters
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        self.theta = np.random.randn(X_b.shape[1])
        prev_loss = float('inf')
        
        for it in range(self.max_iters):
            z = X_b.dot(self.theta)
            predictions = self._sigmoid(z)
            error = predictions - y
            
            # L2 regularization
            reg_term = (self.lambda_/len(X)) * self.theta
            gradients = X_b.T.dot(error)/len(X) + reg_term
            
            # Calculate loss with regularization
            loss = -y.dot(np.log(predictions)) - ((1-y).dot(np.log(1-predictions)))
            loss = loss/len(y) + (self.lambda_/(2*len(y)))*np.sum(self.theta**2)
            
            if abs(prev_loss - loss) < self.tol:
                break
                
            self.theta -= self.alpha * gradients
            prev_loss = loss
            
        return self
    
    def predict_proba(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(X_b.dot(self.theta))

# -----------------------------
# 4. Enhanced CNN with Data Augmentation
# -----------------------------
def build_cnn(input_shape):
    model = Sequential([
        # Data augmentation layers
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),

        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Dropout(0.25),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

def load_images(property_ids, image_folder="property_images"):
    images = []
    valid_ids = []
    
    for pid in property_ids:
        img_path = os.path.join(image_folder, f"{pid}.jpg")
        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(224, 224), color_mode='rgb')
            img_array = tf.keras.preprocessing.image.img_to_array(img)/255.0
            images.append(img_array)
            valid_ids.append(pid)
            
    return np.array(images), valid_ids

# -----------------------------
# 5. Robust Model Fusion
# -----------------------------
class InvestmentEnsemble:
    def __init__(self):
        self.meta_model = LogisticRegression()
        self.price_scaler = RobustScaler()
        
    def fit(self, price_preds, logit_probas, cnn_probas, y_true):
        # Normalize price predictions
        price_norm = self.price_scaler.fit_transform(np.array(price_preds).reshape(-1,1))
        
        # Create meta-features matrix
        X_meta = np.column_stack([price_norm, logit_probas, cnn_probas])
        
        # Train meta-model
        self.meta_model.fit(X_meta, y_true)
        
    def predict(self, price_preds, logit_probas, cnn_probas):
        price_norm = self.price_scaler.transform(np.array(price_preds).reshape(-1,1))
        X_meta = np.column_stack([price_norm, logit_probas, cnn_probas])
        return self.meta_model.predict_proba(X_meta)[:,1]

# -----------------------------
# Main Execution Flow
# -----------------------------
if __name__ == "__main__":
    # Load data with property IDs
    (X_train, X_test, y_train_price, y_test_price,
     y_train_invest, y_test_invest,
     train_ids, test_ids) = load_data()
    
    # Initialize ensemble
    ensemble = InvestmentEnsemble()
    
    # K-Fold Cross Validation for out-of-fold predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {
        'price': np.zeros_like(y_train_price),
        'logit': np.zeros(len(y_train_invest)),
        'cnn': np.zeros(len(y_train_invest))
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining Fold {fold+1}/5")
        
        # Split data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr_price, y_val_price = y_train_price[train_idx], y_train_price[val_idx]
        y_tr_invest, y_val_invest = y_train_invest[train_idx], y_train_invest[val_idx]
        train_fold_ids = train_ids[train_idx]
        val_fold_ids = train_ids[val_idx]
        
        # 1. Train Linear Regression
        lr_model = LinearRegressionGD(alpha=0.01).fit(X_tr, y_tr_price)
        oof_predictions['price'][val_idx] = lr_model.predict(X_val)
        
        # 2. Train Logistic Regression
        logit_model = LogisticRegressionGD(lambda_=0.1).fit(X_tr, y_tr_invest)
        oof_predictions['logit'][val_idx] = logit_model.predict_proba(X_val)
        
        # 3. Train CNN with Images
        train_images, train_img_ids = load_images(train_fold_ids)
        val_images, val_img_ids = load_images(val_fold_ids)
        
        # Align indices for validation images
        val_img_idx = [np.where(val_fold_ids == vid)[0][0] for vid in val_img_ids]
        cnn_model = build_cnn((224, 224, 3))
        early_stop = EarlyStopping(patience=3, restore_best_weights=True)
        
        cnn_model.fit(
            train_images, y_tr_invest[train_img_ids],
            validation_data=(val_images, y_val_invest[val_img_idx]),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        oof_predictions['cnn'][val_idx[val_img_idx]] = cnn_model.predict(val_images).flatten()
    
    # Train final meta-model on out-of-fold predictions
    ensemble.fit(oof_predictions['price'], 
                oof_predictions['logit'], 
                oof_predictions['cnn'], 
                y_train_invest)
    
    # Final evaluation on test set
    # 1. Train final base models on full training data
    final_lr = LinearRegressionGD(alpha=0.01).fit(X_train, y_train_price)
    test_price_pred = final_lr.predict(X_test)
    
    final_logit = LogisticRegressionGD(lambda_=0.1).fit(X_train, y_train_invest)
    test_logit_proba = final_logit.predict_proba(X_test)
    
    # Load and predict with CNN
    test_images, test_img_ids = load_images(test_ids)
    test_img_idx = [np.where(test_ids == tid)[0][0] for tid in test_img_ids]
    final_cnn = build_cnn((224, 224, 3))
    final_cnn.fit(
        load_images(train_ids)[0], y_train_invest,
        epochs=30,
        batch_size=32,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0
    )
    test_cnn_proba = final_cnn.predict(test_images).flatten()
    
    # Align predictions with original test indices
    aligned_cnn_proba = np.zeros(len(test_ids))
    aligned_cnn_proba[test_img_idx] = test_cnn_proba
    
    # final predictions
    final_proba = ensemble.predict(test_price_pred, test_logit_proba, aligned_cnn_proba)
    
    # Evaluation
    print("\nFinal Ensemble Performance:")
    print(f"AUC: {roc_auc_score(y_test_invest, final_proba):.2f}")
    print(f"Accuracy: {accuracy_score(y_test_invest, (final_proba > 0.5).astype(int)):.2f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_price, final_proba, c=y_test_invest, cmap='coolwarm', alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Investment Probability")
    plt.title("Price vs Investment Probability")
    plt.colorbar(label='True Investment Class')
    plt.show()
# =============================================================================
# Conclusion:
# This project integrates real-world data from a real estate price prediction dataset
# and property images to build a comprehensive investment analysis system.
# =============================================================================
