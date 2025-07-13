import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import pickle
import os

# --- Determinism for reproducibility ---
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# -----------------------------
# 1. Enhanced Data Loading & Preprocessing
# -----------------------------
def load_data(test_size=0.2):
    df = pd.read_csv("real_estate_price_prediction.csv")
    assert df.shape[0] > 0, "No data loaded"
    assert 'property_id' in df.columns, "Dataset must contain 'property_id' column"
    
    features = ['Area', 'Floor', 'Num_Bedrooms', 'Num_Bathrooms', 'Property_Age', 'Proximity']
    target = 'Price'
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
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
            y_train_invest.values, y_test_invest.values, # .values is crucial
            train_df['property_id'].values, test_df['property_id'].values)

# -----------------------------
# 2. Custom Model Implementations
# -----------------------------
class LinearRegressionGD:
    def __init__(self, alpha=0.01, tol=1e-4, max_iters=10000):
        self.alpha, self.tol, self.max_iters = alpha, tol, max_iters
    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.random.randn(X_b.shape[1])
        prev_loss = float('inf')
        for _ in range(self.max_iters):
            predictions = X_b.dot(self.theta)
            error = predictions - y
            loss = np.mean(error**2)
            if abs(prev_loss - loss) < self.tol: break
            gradients = 2/len(X) * X_b.T.dot(error)
            self.theta -= self.alpha * gradients
            prev_loss = loss
        return self
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(self.theta)

class LogisticRegressionGD:
    def __init__(self, alpha=0.1, lambda_=0.1, tol=1e-4, max_iters=5000):
        self.alpha, self.lambda_, self.tol, self.max_iters = alpha, lambda_, tol, max_iters
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.theta = np.random.randn(X_b.shape[1])
        prev_loss = float('inf')
        m = len(y)
        for _ in range(self.max_iters):
            z = X_b.dot(self.theta)
            predictions = self._sigmoid(z)
            error = predictions - y
            reg_term = (self.lambda_/m) * np.r_[[0], self.theta[1:]] # Don't regularize bias term
            gradients = (X_b.T.dot(error) / m) + reg_term
            loss = -np.mean(y * np.log(predictions) + (1-y) * np.log(1-predictions)) + (self.lambda_ / (2*m)) * np.sum(self.theta[1:]**2)
            if abs(prev_loss - loss) < self.tol: break
            self.theta -= self.alpha * gradients
            prev_loss = loss
        return self
    def predict_proba(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return self._sigmoid(X_b.dot(self.theta))

# -----------------------------
# 4. CNN and Image Loading 
# -----------------------------
def build_cnn(input_shape):
    model = Sequential([
        RandomFlip("horizontal_and_vertical", input_shape=input_shape),
        RandomRotation(0.2),
        Conv2D(32, (3,3), activation='relu'), MaxPooling2D((2,2)), Dropout(0.25),
        Conv2D(64, (3,3), activation='relu'), MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def load_images(property_ids, image_folder="property_images"):
    images, valid_ids = [], []
    for pid in property_ids:
        img_path = os.path.join(image_folder, f"{pid}.jpg")
        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            valid_ids.append(pid)
    return np.array(images), valid_ids

# -----------------------------
# 5. Ensemble Model 
# -----------------------------
class InvestmentEnsemble:
    def __init__(self):
        self.meta_model = LogisticRegression()
        self.price_scaler = RobustScaler()
    def fit(self, price_preds, logit_probas, cnn_probas, y_true):
        price_norm = self.price_scaler.fit_transform(np.array(price_preds).reshape(-1,1))
        X_meta = np.column_stack([price_norm, logit_probas, cnn_probas])
        self.meta_model.fit(X_meta, y_true)
    def predict(self, price_preds, logit_probas, cnn_probas):
        price_norm = self.price_scaler.transform(np.array(price_preds).reshape(-1,1))
        X_meta = np.column_stack([price_norm, logit_probas, cnn_probas])
        return self.meta_model.predict_proba(X_meta)[:,1]

# -----------------------------
# --- MAIN EXECUTION FLOW ---
# (This is the complete and corrected version with detailed evaluation)
# -----------------------------
if __name__ == "__main__":
    # 1. Load Data
    (X_train, X_test, y_train_price, y_test_price,
     y_train_invest, y_test_invest,
     train_ids, test_ids) = load_data()
    
    # 2. K-Fold CV for Meta-Model Training
    print("--- Starting 5-Fold Cross-Validation for Meta-Model ---")
    ensemble = InvestmentEnsemble()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = {
        'price': np.zeros_like(y_train_price, dtype=float),
        'logit': np.zeros_like(y_train_invest, dtype=float),
        'cnn': np.zeros_like(y_train_invest, dtype=float)
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining Fold {fold+1}/5...")
        # Data splits for this fold
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr_price, y_val_price = y_train_price[train_idx], y_train_price[val_idx]
        y_tr_invest, y_val_invest = y_train_invest[train_idx], y_train_invest[val_idx]
        train_fold_ids, val_fold_ids = train_ids[train_idx], train_ids[val_idx]
        
        # Train base models and get out-of-fold predictions
        lr_model = LinearRegressionGD().fit(X_tr, y_tr_price)
        oof_predictions['price'][val_idx] = lr_model.predict(X_val)
        
        logit_model = LogisticRegressionGD().fit(X_tr, y_tr_invest)
        oof_predictions['logit'][val_idx] = logit_model.predict_proba(X_val)
        
        # CNN training and prediction
        train_images, train_img_ids = load_images(train_fold_ids)
        val_images, val_img_ids = load_images(val_fold_ids)
        
        if len(train_images) > 0 and len(val_images) > 0:
            train_img_indices = [np.where(train_fold_ids == pid)[0][0] for pid in train_img_ids]
            y_tr_invest_for_cnn = y_tr_invest[train_img_indices]
            
            val_img_indices = [np.where(val_fold_ids == vid)[0][0] for vid in val_img_ids]
            y_val_invest_for_cnn = y_val_invest[val_img_indices]

            cnn_model = build_cnn((224, 224, 3))
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            cnn_model.fit(train_images, y_tr_invest_for_cnn, validation_data=(val_images, y_val_invest_for_cnn),
                          epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)
            
            val_preds = cnn_model.predict(val_images).flatten()
            oof_predictions['cnn'][val_idx[val_img_indices]] = val_preds
    
    # 3. Train Final Meta-Model
    print("\n--- Training Final Ensemble Meta-Model ---")
    ensemble.fit(oof_predictions['price'], oof_predictions['logit'], oof_predictions['cnn'], y_train_invest)
    
    # 4. Final Evaluation on Test Set
    print("\n--- Evaluating Base Models and Ensemble on Test Set ---")
    final_lr = LinearRegressionGD().fit(X_train, y_train_price)
    test_price_pred = final_lr.predict(X_test)
    
    final_logit = LogisticRegressionGD().fit(X_train, y_train_invest)
    test_logit_proba = final_logit.predict_proba(X_test)

    print("Training final CNN on all training data...")
    all_train_images, all_train_ids = load_images(train_ids)
    all_train_labels_idx = [np.where(train_ids == pid)[0][0] for pid in all_train_ids]
    all_train_labels = y_train_invest[all_train_labels_idx]

    final_cnn = build_cnn((224, 224, 3))
    final_cnn.fit(all_train_images, all_train_labels, epochs=15, batch_size=32, verbose=0) 
    
    test_images, test_img_ids = load_images(test_ids)
    test_img_idx = [np.where(test_ids == tid)[0][0] for tid in test_img_ids]
    
    aligned_cnn_proba = np.zeros(len(test_ids))
    if len(test_images) > 0:
        cnn_preds_for_test = final_cnn.predict(test_images).flatten()
        aligned_cnn_proba[test_img_idx] = cnn_preds_for_test

    final_proba = ensemble.predict(test_price_pred, test_logit_proba, aligned_cnn_proba)
    
    # --- METRICS CALCULATION AND REPORTING ---
    lr_rmse = np.sqrt(mean_squared_error(y_test_price, test_price_pred))
    logit_auc = roc_auc_score(y_test_invest, test_logit_proba)
    logit_acc = accuracy_score(y_test_invest, (test_logit_proba > 0.5).astype(int))

    y_test_invest_for_cnn = y_test_invest[test_img_idx]
    if len(test_images) > 0:
        cnn_auc = roc_auc_score(y_test_invest_for_cnn, cnn_preds_for_test)
        cnn_acc = accuracy_score(y_test_invest_for_cnn, (cnn_preds_for_test > 0.5).astype(int))
    else:
        cnn_auc, cnn_acc = 0, 0
    
    ensemble_auc = roc_auc_score(y_test_invest, final_proba)
    ensemble_acc = accuracy_score(y_test_invest, (final_proba > 0.5).astype(int))

    print("\n" + "="*50)
    print("Final Model Performance Comparison".center(50))
    print("="*50)
    print(f"Linear Regression RMSE (Price Prediction): ${lr_rmse:,.2f}")
    print("-" * 50)
    print("Logistic Regression (Structured Data Only):")
    print(f"  - AUC:      {logit_auc:.4f}")
    print(f"  - Accuracy: {logit_acc:.4f}")
    print("-" * 50)
    print(f"CNN (Image Data Only, on {len(test_images)} properties):")
    print(f"  - AUC:      {cnn_auc:.4f}")
    print(f"  - Accuracy: {cnn_acc:.4f}")
    print("-" * 50)
    print("ENSEMBLE MODEL (Combined):")
    print(f"  - AUC:      {ensemble_auc:.4f}")
    print(f"  - Accuracy: {ensemble_acc:.4f}")
    print("="*50)

    best_single_model_auc = max(logit_auc, cnn_auc)
    if best_single_model_auc > 0:
        improvement = ((ensemble_auc - best_single_model_auc) / best_single_model_auc) * 100
        print(f"\nEnsemble shows a {improvement:.2f}% improvement in AUC over the best single model.")
    
    # 5. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_price, final_proba, c=y_test_invest, cmap='coolwarm', alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Investment Probability")
    plt.title("Price vs Investment Probability")
    plt.colorbar(label='True Investment Class')
    plt.show()

    # 6. Save final models for production
    print("\n--- Saving final models for production... ---")
    final_scaler = StandardScaler().fit(X_train)
    with open('prod_scaler.pkl', 'wb') as f: pickle.dump(final_scaler, f)
    with open('prod_linear_regression.pkl', 'wb') as f: pickle.dump(final_lr.theta, f)
    with open('prod_logistic_regression.pkl', 'wb') as f: pickle.dump(final_logit.theta, f)
    final_cnn.save('prod_cnn_model.keras')
    with open('prod_ensemble.pkl', 'wb') as f: pickle.dump(ensemble, f)
    print("\nModels saved successfully.")
