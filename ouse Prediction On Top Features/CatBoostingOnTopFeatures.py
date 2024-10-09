import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv(r'D:\MS Ai900\class2\Week 2 Course work\corr.csv')

features_to_remove = [
   'GrLivArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'TotalBsmtSF',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'YearBuilt', 'YrSold', 'YearRemodAdd', 'OverallCond', 'OverallQual',
    'FullBath', 'HalfBath'
]

# Remove the specified features from the dataset
df = df.drop(columns=features_to_remove)

# Correlation analysis
corelated_df = df.corr()
print(corelated_df.head())

corr = corelated_df['SalePrice']
print(corr.head())

# Select top features
top_features = corr.abs().sort_values(ascending=False).head(25).index.tolist()
print(top_features)
top_features.remove('SalePrice')

# Features and target variable
X = df.drop(columns=['Id', 'SalePrice'])
Y = df['SalePrice']
X_top_Features = X[top_features]

ab2 = X_top_Features
ab2.to_csv(r'D:\MS Ai900\class2\Week 2 Course work\topfeatures2.csv', index=False)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_top_Features, Y, test_size=0.3, random_state=42)

# RandomForestRegressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train, Y_train)
Prediction_rf = model_rf.predict(X_test)

# RandomForest evaluation metrics
rmse_rf = np.sqrt(((np.log(Prediction_rf + 1) - np.log(Y_test + 1)) ** 2).mean())
print("RMSE (RandomForest):", rmse_rf)

mae_rf = mean_absolute_error(Y_test, Prediction_rf)
print("MAE (RandomForest):", mae_rf)

r2_rf = r2_score(Y_test, Prediction_rf)
print("R² score (RandomForest):", r2_rf)

# LGBMRegressor
model_lgbm = LGBMRegressor()
model_lgbm.fit(X_train, Y_train)
Prediction_lgbm = model_lgbm.predict(X_test)

# LGBM evaluation metrics
rmse_lgbm = np.sqrt(((np.log(Prediction_lgbm + 1) - np.log(Y_test + 1)) ** 2).mean())
print("RMSE (LGBM):", rmse_lgbm)

mae_lgbm = mean_absolute_error(Y_test, Prediction_lgbm)
print("MAE (LGBM):", mae_lgbm)

r2_lgbm = r2_score(Y_test, Prediction_lgbm)
print("R² score (LGBM):", r2_lgbm)

# CatBoostRegressor
model_catboost = CatBoostRegressor(learning_rate=0.1, depth=7, iterations=215, silent=True, l2_leaf_reg=1)
model_catboost.fit(X_train, Y_train)
Prediction_catboost = model_catboost.predict(X_test)

# CatBoost evaluation metrics
rmse_catboost = np.sqrt(((np.log(Prediction_catboost + 1) - np.log(Y_test + 1)) ** 2).mean())
print("RMSE (CatBoost):", rmse_catboost)

mae_catboost = mean_absolute_error(Y_test, Prediction_catboost)
print("MAE (CatBoost):", mae_catboost)

r2_catboost = r2_score(Y_test, Prediction_catboost)
print("R² score (CatBoost):", r2_catboost)

# Print all RMSE values together
print("\nSummary of RMSE Values:")
print(f"RandomForest RMSE: {rmse_rf}")
print(f"LGBM RMSE: {rmse_lgbm}")
print(f"CatBoost RMSE: {rmse_catboost}")

# Save predictions to CSV
ab = pd.DataFrame({
    'Actual': Y_test,
    'RandomForest_Prediction': Prediction_rf,
    'LGBM_Prediction': Prediction_lgbm,
    'CatBoost_Prediction': Prediction_catboost
})
ab.to_csv(r'D:\MS Ai900\class2\Week 2 Course work\Predictions_Comparison.csv', index=False)
