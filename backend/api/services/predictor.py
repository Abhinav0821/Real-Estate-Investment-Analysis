import numpy as np
import pickle
import tensorflow as tf
from django.conf import settings
import os

# --- Re-create your custom model classes ---
# These are needed so pickle can reconstruct the objects
class LinearRegressionGD:
    def __init__(self, theta): self.theta = theta
    def predict(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return X_b.dot(self.theta)

class LogisticRegressionGD:
    def __init__(self, theta): self.theta = theta
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def predict_proba(self, X):
        X_b = np.hstack([np.ones((len(X), 1)), X])
        return self._sigmoid(X_b.dot(self.theta))

# We need the ensemble class for loading
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
class InvestmentEnsemble:
    def __init__(self):
        self.meta_model = LogisticRegression()
        self.price_scaler = RobustScaler()
    # The rest of the class definition...
    # (Copy the class from your original script)
    def fit(self, price_preds, logit_probas, cnn_probas, y_true):
        price_norm = self.price_scaler.fit_transform(np.array(price_preds).reshape(-1,1))
        X_meta = np.column_stack([price_norm, logit_probas, cnn_probas])
        self.meta_model.fit(X_meta, y_true)
    def predict(self, price_preds, logit_probas, cnn_probas):
        price_norm = self.price_scaler.transform(np.array(price_preds).reshape(-1,1))
        X_meta = np.column_stack([price_norm, logit_probas, cnn_probas])
        return self.meta_model.predict_proba(X_meta)[:,1]

# --- Prediction Service Singleton ---
class PredictionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        model_dir = os.path.join(settings.BASE_DIR, 'api/ml_models')

        with open(os.path.join(model_dir, 'prod_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(os.path.join(model_dir, 'prod_linear_regression.pkl'), 'rb') as f:
            theta = pickle.load(f)
            self.lr_model = LinearRegressionGD(theta=theta)

        with open(os.path.join(model_dir, 'prod_logistic_regression.pkl'), 'rb') as f:
            theta = pickle.load(f)
            self.logit_model = LogisticRegressionGD(theta=theta)
            
        with open(os.path.join(model_dir, 'prod_ensemble.pkl'), 'rb') as f:
            self.ensemble_model = pickle.load(f)

        self.cnn_model = tf.keras.models.load_model(os.path.join(model_dir, 'prod_cnn_model.keras'))
        print("✅ All ML models loaded successfully.")

    def predict(self, input_data, image_file):
        # 1. Prepare Structured Data
        features = ['Area', 'Floor', 'Num_Bedrooms', 'Num_Bathrooms', 'Property_Age', 'Proximity']
        structured_values = np.array([[input_data[f] for f in features]])
        scaled_features = self.scaler.transform(structured_values)

        # 2. Prepare Image Data
        img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0) # Create a batch of 1

        # 3. Get Base Model Predictions
        price_pred = self.lr_model.predict(scaled_features)
        logit_proba = self.logit_model.predict_proba(scaled_features)
        cnn_proba = self.cnn_model.predict(img_batch).flatten()

        # 4. Get Final Ensemble Prediction
        final_proba = self.ensemble_model.predict(price_pred, logit_proba, cnn_proba)

        return final_proba[0]

# Instantiate the service so models are loaded on server start
prediction_service = PredictionService()