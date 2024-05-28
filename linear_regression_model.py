import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def encode_and_prepare_features(data):
    # Prepare the features and encode categorical data
    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(), ['room_type', 'neighbourhood_cleansed'])],
        remainder='passthrough'
    )
    features_encoded = ct.fit_transform(data[['room_type', 'neighbourhood_cleansed', 'accommodates', 'number_of_reviews']])
    return features_encoded, data['price']

def train_linear_regression(X_train, y_train):
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Evaluate the model using MSE and R^2 Score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")
