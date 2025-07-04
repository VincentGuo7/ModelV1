import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys


## Uses previously prepared Data
## Uses previously trained Model



######### Getting Feature titles of the data ################
def get_titles(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)

    print(f"Columns in dataset: {df.columns.tolist()}")

    non_feature_cols = ['date', 'lat', 'lon', 'land_sea_mask', 'orography', 'latitude']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    return feature_cols




###### Splitting Data #######
# def train_val_test_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
def data_split(X, y, train_size=0.8, test_size=0.2, random_state=42):

    assert abs(train_size + test_size - 1.0) < 1e-6, "Splits must add up to 1."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # val_ratio = val_size / (train_size + val_size)
    
    # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)
    
    print(f"\nSize of training dataset: {len(y_train)}\n")

    return X_train, X_test, y_train, y_test
    


###### Flattening Data #######
class SequenceFlattener(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Save shape information to apply same reshape during transform
        self.n_timesteps = X.shape[1]
        self.n_features = X.shape[2]
        return self

    def transform(self, X):
        n_samples = X.shape[0]  # Infer sample size here
        return X.reshape(n_samples, self.n_timesteps * self.n_features)
    

    

####### Creating RFR with customised hyperparameters ########
class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 n_estimators=288,
                 max_features='sqrt',
                 max_samples=0.498877047886193,
                 max_depth=29,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.23464427
                 ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        # self.model = None


    def fit(self, X, y):
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_samples=self.max_samples,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease
        )
        self.model_.fit(X, y)
        return self


    def predict(self, X):
        return self.model_.predict(X)



############### Defining the pipeline ####################
def create_pipeline(model):

    # No categorical features to encode in this context, but if there were:

    pipeline = Pipeline(steps=[
        ('preprocessor', SequenceFlattener()),
        ('regressor', model)
    ])

    return pipeline



######## Evaluating the model ##############
def evaluate_model(y_pred, y_test, lat_values, feature_titles):

    print(f"\n------------ Evaluation Metrics ------------")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")


    # Compute weights: inverse squared range per feature
    ranges = y_test.max(axis=0) - y_test.min(axis=0)
    # Avoid divide by zero
    ranges = np.where(ranges == 0, 1e-8, ranges)
    weights = 1.0 / (ranges ** 2)

    # y_true, y_pred shape: (n_samples, n_features)
    # weights shape: (n_features,)
    se = (y_test - y_pred) ** 2
    weighted_se = se * weights
    print(f"WRMSE: {np.sqrt(np.mean(np.sum(weighted_se, axis=1))):.4f}")


    # Normalize weights
    w_lat = np.cos(np.deg2rad(lat_values))
    w_lat = w_lat / w_lat.mean()

    # Apply weights (assume y_test/pred shape = [num_samples, num_features])
    # w_lat should be [num_samples]
    w_lat = w_lat.reshape(-1, 1)



    print(f"\n---------- Per Feature Evaluation ----------")
    
    mse_values = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse_values = np.sqrt(mse_values)

    errors = (y_test - y_pred) ** 2
    weighted_errors = errors * w_lat

    w_mse_values = np.mean(weighted_errors, axis=0)
    w_rmse_values = np.sqrt(w_mse_values)


    ranges = y_test.max(axis=0) - y_test.min(axis=0)
    ranges = np.where(ranges == 0, 1e-8, ranges)  
    nrmse_range = rmse_values / ranges


    std_dev = y_test.std(axis=0)
    std_dev = np.where(std_dev == 0, 1e-8, std_dev)
    nrmse_std = rmse_values / std_dev


    mean_y = np.nanmean(y_test, axis=0)
    mean_y_safe = np.where(np.abs(mean_y) < 1e-8, 1e-8, mean_y)  # avoid zero mean
    nrmse_mean = rmse_values / mean_y_safe


    ## Percentage Error Calculation ##
    safe_y_test = np.where(y_test == 0, np.nan, y_test)
    percentage_errors = np.abs((y_test - y_pred) / np.abs(safe_y_test)) * 100

    # Compute average percentage error per feature
    avg_percentage_error = np.nanmean(percentage_errors, axis=0)

    # Suppress error% for features with very small mean values
    mean_y = np.nanmean(safe_y_test, axis=0)
    error_threshold = 1  # or adjust based on your data's scale

    # Mask error% where mean target is too small
    avg_percentage_error = np.where(np.abs(mean_y) < error_threshold, np.nan, avg_percentage_error)


    ### Symmetric Mean Absolute Percentage Error ########
    denominator = (np.abs(y_test) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.abs(y_test - y_pred) / denominator * 100
    avg_smape = np.nanmean(smape, axis=0)


    metrics_table = pd.DataFrame({
        'Feature': feature_titles,
        'MSE': [f"{mse:.4f}" for mse in mse_values],
        'RMSE': [f"{rmse:.4f}" for rmse in rmse_values],
        'NRMSE_range': [f"{nrmse:.4f}" for nrmse in nrmse_range],
        'NRMSE_std': [f"{nrmse:.4f}" for nrmse in nrmse_std],
        'NRMSE_mean': [f"{nrmse:.4f}" for nrmse in nrmse_mean],
        'Error%': [f"{perror:.2f}" if not np.isnan(perror) else "--" for perror in avg_percentage_error],
        'sMAPE%': [f"{val:.2f}" for val in avg_smape],
        'Mean': [f"{mean:.2f}" for mean in mean_y],
        'W_MSE': [f"{w_mse:.6f}" for w_mse in w_mse_values],
        'W_RMSE': [f"{w_rmse:.6f}" for w_rmse in w_rmse_values],
    })

    print(metrics_table.to_string(index=False))




    # print(f"\n---------- Target Variables Evaluation ----------\n")

    # # Variables of interest for ClimaX output (adapt to your actual column names)
    # target_variables = [
    #     "geopotential_500", "temperature_850", "2m_temperature",
    #     "10m_u_component_of_wind", "10m_v_component_of_wind"
    # ]

    # se = (y_test - y_pred) ** 2
    # w_se = se * w_lat


    # for var in target_variables:
    #     idx = feature_titles.index(var)
    #     w_mse = np.mean(w_se[:, idx])
    #     w_rmse = np.sqrt(w_mse)
    #     print(f"w_mse_{var}: {w_mse:.6f}")
    #     print(f"w_rmse_{var}: {w_rmse:.6f}")

    # print(f"---------------------------------------------------\n")




############## ACTION CODE #################

X = np.load("X_data_final.npy")
y_full = np.load("y_data_final.npy")

print("\n✅ Data has been loaded into Arrays! \n\n")

X_train, X_test, y_train_full, y_test_full = data_split(X,y_full)

lat_values = y_test_full[:, 0]
y_test = y_test_full[:, 5:]

pipeline = joblib.load('trained_RF_model_rmse_optimal.pkl')

print("\n Predicting results! \n\n")
y_pred = pipeline.predict(X_test)

print("\n Evaluating model! \n\n")

feature_titles = get_titles('climax_processed_NA.parquet')

evaluate_model(y_pred, y_test, lat_values, feature_titles)
