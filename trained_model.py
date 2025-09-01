import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\2015_extended.csv", encoding="latin1")
# Drop non-numeric columns (Country, Region)
df = df.drop(["Country", "Region"], axis=1)

# Define features and target
X = df.drop(["Happiness Score"], axis=1)
y = df["Happiness Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SVR": SVR(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "XGBoost": XGBRegressor()
}

trained_models = {}

# Train and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name}: R2 Score = {score:.4f}, RMSE = {rmse:.4f}")
    trained_models[name] = model

# Save all models and feature names
joblib.dump({"models": trained_models, "features": X.columns.tolist()}, "all_regression_models.pkl")
print("\n✅ All models trained and saved successfully!")
