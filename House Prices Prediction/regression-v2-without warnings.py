import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, LabelBinarizer
from category_encoders import TargetEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import optuna
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_df = pd.read_csv(r"D:\MS Ai900\class2\Week 2 Course work\train.csv")
train_df.head()

train_df.columns

discrete_features = train_df.select_dtypes(include=['int', 'float'])
categorical_features = train_df.select_dtypes(include=['object'])
print(f'Discrete features are:... {discrete_features.columns}\n\n Categorical features are: {categorical_features.columns}')

columns_with_null_values = train_df.columns[train_df.isnull().any()]
columns_with_null_values

discrete_features_with_nulls = [col for col in columns_with_null_values if pd.api.types.is_float_dtype(train_df[col])]
categorical_features_with_nulls = [col for col in columns_with_null_values if pd.api.types.is_object_dtype(train_df[col])]
print(f"Discrete features that contain null values are: {discrete_features_with_nulls}")
print(f"Categorical features that contain null values are: {categorical_features_with_nulls}")

plt.figure(figsize=(15,8))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.show()

sns.relplot(x="SaleCondition", y="YrSold", data=train_df);

sns.lmplot(x="YrSold", y="SalePrice", hue="SaleCondition", data=train_df);

sns.relplot(x="YrSold", y="SalePrice", data=train_df);

# Add a value of 0 whenever there is a null value in the MasVnrType column
train_df.loc[train_df['MasVnrType'].isna(), 'MasVnrArea'] = 0

# Calculate the average LotFrontage within each MSZoning group
average_lot_frontage = train_df.groupby('MSZoning')['LotFrontage'].transform('mean')

# Fill null values in LotFrontage with the group-wise averages
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(average_lot_frontage)

# Add a value of No whenever there is a null value in the BsmtExposure column
condition = (train_df['BsmtCond'].notna()) & (train_df['BsmtExposure'].isna())
train_df.loc[condition, 'BsmtExposure'] = 'No'

# Give a value of FuseF to null row in Electrical because based on the SalePrice, it is a fair amount for a FuseF
train_df.loc[train_df['Electrical'].isna(), 'Electrical'] = 'FuseF'

train_df['BsmtQual'].fillna('None', inplace=True)
train_df['BsmtCond'].fillna('None', inplace=True)
train_df['BsmtExposure'].fillna('None', inplace=True)
train_df['BsmtFinType1'].fillna('None', inplace=True)
train_df['BsmtFinType2'].fillna('None', inplace=True)
train_df['FireplaceQu'].fillna('None', inplace=True)
train_df['GarageFinish'].fillna('None', inplace=True)
train_df['GarageQual'].fillna('None', inplace=True)
train_df['GarageCond'].fillna('None', inplace=True)
train_df['PoolQC'].fillna('None', inplace=True)
train_df['ExterQual'].fillna('None', inplace=True)
train_df['Fence'].fillna('None', inplace=True)
train_df['HeatingQC'].fillna('None', inplace=True)
train_df['GarageYrBlt'].fillna(train_df['YearBuilt'], inplace=True)

bsmt_qual_ordered_categories = [['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]
ordinal_encoder = OrdinalEncoder(categories=bsmt_qual_ordered_categories)
train_df['BsmtQual'] = ordinal_encoder.fit_transform(train_df[['BsmtQual']])
train_df['BsmtCond'] = ordinal_encoder.fit_transform(train_df[['BsmtCond']])

bsmt_exposure_ordered_categories = [['None','No', 'Mn', 'Av', 'Gd']]
ordinal_encoder = OrdinalEncoder(categories=bsmt_exposure_ordered_categories)
train_df['BsmtExposure'] = ordinal_encoder.fit_transform(train_df[['BsmtExposure']])

bsmt_fin_type_1_categories = [['None','Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']]
ordinal_encoder = OrdinalEncoder(categories=bsmt_fin_type_1_categories)
train_df['BsmtFinType1'] = ordinal_encoder.fit_transform(train_df[['BsmtFinType1']])
train_df['BsmtFinType2'] = ordinal_encoder.fit_transform(train_df[['BsmtFinType2']])

electrical_categories = [['FuseP', 'FuseF', 'Mix', 'FuseA', 'SBrkr']]
ordinal_encoder = OrdinalEncoder(categories=electrical_categories)
train_df['Electrical'] = ordinal_encoder.fit_transform(train_df[['Electrical']])

fireplace_quality_categories = [['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]
ordinal_encoder = OrdinalEncoder(categories=fireplace_quality_categories)
train_df['FireplaceQu'] = ordinal_encoder.fit_transform(train_df[['FireplaceQu']])
train_df['HeatingQC'] = ordinal_encoder.fit_transform(train_df[['HeatingQC']])

garage_finish_categories = [['None', 'Unf', 'RFn', 'Fin']]
ordinal_encoder = OrdinalEncoder(categories=garage_finish_categories)
train_df['GarageFinish'] = ordinal_encoder.fit_transform(train_df[['GarageFinish']])

garage_quality_categories = [['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']]
ordinal_encoder = OrdinalEncoder(categories=garage_quality_categories)
train_df['GarageQual'] = ordinal_encoder.fit_transform(train_df[['GarageQual']])
train_df['GarageCond'] = ordinal_encoder.fit_transform(train_df[['GarageCond']])
train_df['KitchenQual'] = ordinal_encoder.fit_transform(train_df[['KitchenQual']])

pool_quality_categories = [['None', 'Fa', 'TA', 'Gd', 'Ex']]
ordinal_encoder = OrdinalEncoder(categories=pool_quality_categories)
train_df['PoolQC'] = ordinal_encoder.fit_transform(train_df[['PoolQC']])

fence_quality_categories = [['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']]
ordinal_encoder = OrdinalEncoder(categories=fence_quality_categories)
train_df['Fence'] = ordinal_encoder.fit_transform(train_df[['Fence']])

lb = LabelBinarizer()
train_df['Street'] = lb.fit_transform(train_df['Street'])
train_df['CentralAir'] = lb.fit_transform(train_df['CentralAir'])

standard_scaler = StandardScaler()
train_df['OverallArea'] = (
    train_df['GrLivArea'] + train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['LowQualFinSF'] +
    train_df['TotalBsmtSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] + train_df['BsmtUnfSF'] +
    train_df['GarageArea'] +
    train_df['WoodDeckSF'] + train_df['OpenPorchSF'] + train_df['EnclosedPorch'] +
    train_df['3SsnPorch'] + train_df['ScreenPorch'] +
    train_df['PoolArea'] +

    train_df['MiscVal']
)




train_df['OverallArea'] = standard_scaler.fit_transform(train_df[['OverallArea']])

target_encoder = TargetEncoder()
for colname in train_df.select_dtypes(include=["object"]):
    train_df[colname] = target_encoder.fit_transform(train_df[[colname]], train_df['OverallQual'])

train_df.head()

## Feature Engineering

train_df['AgeFromBuilt'] = train_df['YrSold'] - train_df['YearBuilt']
train_df['AgeFromModification'] = train_df['YrSold'] - train_df['YearRemodAdd']

train_df['OverallHouseCondition'] = train_df['OverallCond'] * train_df['OverallQual']
train_df['TotalBasementArea'] = train_df['TotalBsmtSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2'] + train_df['BsmtUnfSF']
train_df['TotalBathrooms'] = train_df['FullBath'] + 0.5 * train_df['HalfBath']
train_df['TotalFinishedBsmtArea'] = train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']



## Features and Target definition

y = train_df['SalePrice']
X = train_df.drop(columns=['SalePrice', 'Id'])

## Train-test split

train_df.to_csv(r'D:\MS Ai900\class2\Week 2 Course work\corr.csv', index=False)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)





def custom_metric(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

# Define the scoring metric
scoring = make_scorer(custom_metric, greater_is_better=False)

## Random Forest Regressor

def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=scoring))

# Initialize Optuna study and optimize
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=50)

print('Best parameters:', study_rf.best_params)

## LightGBM Regressor

def objective_lgbm(trial):
    num_leaves = trial.suggest_int('num_leaves', 20, 3000)
    max_depth = trial.suggest_int('max_depth', 3, 50)
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-1)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    min_child_samples = trial.suggest_int('min_child_samples', 1, 300)
    subsample = trial.suggest_float('subsample', 0.2, 1.0)
    
    model = LGBMRegressor(
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        random_state=42
    )
    
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring=scoring))

# Initialize Optuna study and optimize
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=50)

print('Best parameters:', study_lgbm.best_params)

## Final model training with best parameters from each study

best_rf = RandomForestRegressor(
    **study_rf.best_params,
    random_state=42
)

best_lgbm = LGBMRegressor(
    **study_lgbm.best_params,
    random_state=42
)

# Fitting the models
best_rf.fit(X_train, y_train)
best_lgbm.fit(X_train, y_train)

# Making predictions
rf_preds = best_rf.predict(X_valid)
lgbm_preds = best_lgbm.predict(X_valid)

# Evaluate the model performance
rf_score = custom_metric(y_valid, rf_preds)
lgbm_score = custom_metric(y_valid, lgbm_preds)

print(f"Random Forest Regressor score: {rf_score}")
print(f"LightGBM Regressor score: {lgbm_score}")

