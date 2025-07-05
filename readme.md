# Real Estate Investment Analyzer

![Real Estate Investment Analyzer Screenshot](https://via.placeholder.com/800x450.png?text=Add+A+Screenshot+Of+Your+App+Here)

A full-stack web application that leverages a multi-modal machine learning ensemble to predict the investment potential of real estate properties. The system analyzes both structured property data and property images to provide a comprehensive probability score for high-value investments.

---

## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Training](#model-training)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Key Features

-   **Multi-Modal Analysis:** Combines structured data (Area, Bedrooms, Proximity, etc.) and visual data (property images) for more accurate predictions.
-   **Ensemble Modeling:** Utilizes a robust stacking ensemble (meta-learning) model that combines predictions from three base models for superior performance.
-   **Interactive UI:** A clean, responsive, and user-friendly interface built with React and Material-UI allows users to input property details and receive instant analysis.
-   **Full-Stack Implementation:** Built with a scalable architecture using Django REST Framework for the backend and React for the frontend.
-   **Containerized Deployment:** Fully containerized with Docker and Docker Compose for consistent, cross-platform development and easy deployment.

---

## Tech Stack

-   **Backend:** Django, Django REST Framework
-   **Frontend:** React, Material-UI, Axios
-   **Database:** PostgreSQL
-   **Machine Learning:** TensorFlow/Keras, Scikit-learn, Pandas, NumPy
-   **Deployment:** Docker, Docker Compose

---

## Machine Learning Pipeline

The core of this project is a sophisticated ensemble model designed to mitigate the weaknesses of individual models.

1.  **Data Sources:**
    -   **Structured Data:** Real Estate Price Prediction Dataset from Figshare. [[Link to Dataset]](https://doi.org/10.6084/m9.figshare.26517325.v1)
    -   **Image Data:** A supplementary collection of property images corresponding to the structured dataset.

2.  **Base Models:**
    -   **Linear Regression:** A custom Gradient Descent implementation to predict the absolute price of a property.
    -   **Logistic Regression:** A custom L2-regularized model to predict investment potential based on structured data.
    -   **Convolutional Neural Network (CNN):** A Keras/TensorFlow model with data augmentation to predict investment potential based on property images.

3.  **Ensemble Method (Stacking):**
    -   Out-of-fold (OOF) predictions are generated for the entire training set using 5-fold cross-validation to prevent data leakage.
    -   A `LogisticRegression` meta-model is then trained on these OOF predictions to learn how to best combine the outputs from the three base models.

![ML Pipeline Diagram](https://via.placeholder.com/600x250.png?text=Optional:+Add+a+diagram+of+your+ML+pipeline)

---

## System Architecture

The application follows a classic client-server architecture, containerized for portability.

1.  **React Frontend:** The user interacts with the React application running in their browser.
2.  **API Request:** On form submission, the frontend sends a `multipart/form-data` request containing the structured data and image to the backend API.
3.  **Django Backend:**
    -   Receives and validates the request using Django REST Framework serializers.
    -   The `PredictionService` preprocesses the data (scaling, image resizing).
    -   The service feeds the processed data into the pre-loaded ML models.
    -   The final ensemble model computes the investment probability.
    -   The request and result are logged to the PostgreSQL database.
4.  **API Response:** The final probability score is returned to the frontend.
5.  **Display Result:** The React frontend displays the result to the user in a visual gauge.

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
-   [Git](https://git-scm.com/)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abhinav0821/Real-Estate-Investment-Analysis.git
    cd Real-Estate-Investment-Analysis
    ```

2.  **Prepare the Backend Environment File:**
    Navigate to the `backend` directory and create an environment file for the database credentials.
    ```bash
    cd backend
    cp .env.example .env.dev 
    # Or create .env.dev manually and add the content below
    ```
    Your `.env.dev` should contain:
    ```
    POSTGRES_DB=reip_db
    POSTGRES_USER=user
    POSTGRES_PASSWORD=password
    ```

3.  **Place the ML Models:**
    The pre-trained models are required for the application to run. Due to their size, they are not tracked in Git.
    -   Download the models from: `[Link to your models on Google Drive, Dropbox, etc.]`
    -   Unzip and place the five model files (`prod_scaler.pkl`, `prod_cnn_model.keras`, etc.) into the `backend/api/ml_models/` directory.

4.  **Build and Run with Docker Compose:**
    From the project root directory (`Real-Estate-Investment-Analysis/`), run:
    ```bash
    docker compose up --build
    ```

---

## Usage

-   The **Frontend** will be available at `http://localhost:3000`.
-   The **Backend API** will be running at `http://localhost:8000`.

Open your browser to `http://localhost:3000`, fill in the property details, upload an image, and click "Analyze Investment" to see the prediction.

---

## Model Training

The script and data for training the models are located in the `/training` directory (not included in the main application build). To retrain the models, place the datasets in this directory and run the `REIP.py` script. See the script's internal documentation for details.

---

## Future Improvements

-   **Cloud Deployment:** Deploy the application to a cloud service like AWS, GCP, or Heroku.
-   **Advanced Image Models:** Use a more powerful pre-trained CNN (like EfficientNet or ResNet) for the image analysis task through transfer learning.
-   **Data-Drift Monitoring:** Implement a system to monitor the performance of the live model and trigger retraining when performance degrades.
-   **User Accounts:** Add user authentication to allow users to save and track their past analyses.

---

## Contact

Abhinav Kashyap - [Your LinkedIn Profile URL] - [your.email@example.com]

Project Link: [https://github.com/Abhinav0821/Real-Estate-Investment-Analysis](https://github.com/Abhinav0821/Real-Estate-Investment-Analysis)
