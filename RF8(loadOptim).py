import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from optuna.study import MaxTrialsCallback
from joblib import Parallel, delayed
from optuna.study import Study




## Uses previously prepared Data
## Runs and trains a new Model



######### Getting Feature titles of the data ################
def get_titles(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)

    print(f"Columns in dataset: {df.columns.tolist()}")

    non_feature_cols = ['date', 'lat', 'lon']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    return feature_cols




###### Splitting Data #######
def data_split(X, y, train_size=0.8, test_size=0.2, random_state=42):
    assert abs(train_size + test_size - 1.0) < 1e-6, "Splits must add up to 1."

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
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
    




######## Evaluating the model ##############
def evaluate_model(pred_test, y_test, feature_titles):
    print(f"\n------------ Evaluation Metrics ------------")
    print(f"MAE: {mean_absolute_error(y_test, pred_test):.4f}")
    print(f"MSE: {mean_squared_error(y_test, pred_test):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
    print(f"R²: {r2_score(y_test, pred_test):.4f}")

    print(f"\n---------- Per Feature Evaluation ----------")
    
    mse_values = mean_squared_error(y_test, pred_test, multioutput='raw_values')
    rmse_values = np.sqrt(mse_values)


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
    percentage_errors = np.abs((y_test - pred_test) / np.abs(safe_y_test)) * 100

    # Compute average percentage error per feature
    avg_percentage_error = np.nanmean(percentage_errors, axis=0)

    # Suppress error% for features with very small mean values
    mean_y = np.nanmean(safe_y_test, axis=0)
    error_threshold = 1  # or adjust based on your data's scale

    # Mask error% where mean target is too small
    avg_percentage_error = np.where(np.abs(mean_y) < error_threshold, np.nan, avg_percentage_error)



    ### Symmetric Mean Absolute Percentage Error ########
    denominator = (np.abs(y_test) + np.abs(pred_test)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    smape = np.abs(y_test - pred_test) / denominator * 100
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
    'Mean': [f"{mean:.2f}" for mean in mean_y]
    })

    print(metrics_table.to_string(index=False))
    print(f"--------------------------------------------\n")




############# Optimizing the model ##############


def weighted_rmse(y_true, y_pred):

    #
    # Computes a single RMSE score over all targets, weighted by the inverse squared range
    # of each feature (feature-wise weights).
    #

    # Compute weights: inverse squared range per feature
    ranges = y_true.max(axis=0) - y_true.min(axis=0)
    # Avoid divide by zero
    ranges = np.where(ranges == 0, 1e-8, ranges)
    weights = 1.0 / (ranges ** 2)

    # y_true, y_pred shape: (n_samples, n_features)
    # weights shape: (n_features,)
    se = (y_true - y_pred) ** 2
    weighted_se = se * weights
    return np.sqrt(np.mean(np.sum(weighted_se, axis=1)))



def create_objective(X_train, y_train):

    def objective(trial):
        # Suggest hyperparameters

        print(f"🔁 Starting trial {trial.number}")

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'max_samples': trial.suggest_float('max_samples', 0.3, 0.5),
            'max_depth': trial.suggest_int('max_depth', 15, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.01, 0.25)
        }

        model = CustomRandomForestRegressor(**params)
        pipeline = create_pipeline(model)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        scorer = make_scorer(weighted_rmse, greater_is_better=False)
        
        scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring=scorer, n_jobs=-1)

        print(f"✅ Trial {trial.number} finished with mean Weighted RMSE: {scores.mean():.4f}")

        return -scores.mean()
    
    return objective




####### Creating RFR with customised hyperparameters ########
class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 n_estimators=150,
                 max_features='sqrt',
                 max_samples=0.3,
                 max_depth=30,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_impurity_decrease=0.05
                 ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease


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




############## ACTION CODE #################

feature_titles = get_titles('climax_processed_NA.parquet')
X = np.load("X_data_final.npy")
y_full = np.load("y_data_final.npy")

print("\n✅ Data has been loaded into Arrays! \n\n")

y = y_full[:, 2:]

X_train, X_test, y_train, y_test = data_split(X,y)


print("\n🔍 Starting Bayesian Optimization with Optuna... \n")

study = optuna.load_study(
    study_name="random_forest_optimization_weighted_rmse",
    storage="sqlite:///optuna_study_weighted_rmse.db"
)


# study.enqueue_trial({
#     'n_estimators': 150,
#     'max_features': 'log2',
#     'max_samples': 0.3,
#     'max_depth': 30,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
#     'min_impurity_decrease': 0.05
# })


objective = create_objective(X_train, y_train)

# # Enqueue study
# study.optimize(objective, n_trials=1, show_progress_bar=True, n_jobs=1)


# ##SQL Sequencing shit
# # 🔄 Manual parallelism using Joblib to run parallel suggestions

# def run_trial(study: Study, trial_number: int):
#     trial = study.ask()
#     value = objective(trial)
#     study.tell(trial, value)

# # 🎯 Run parallel trials
# n_trials_parallel = 100
# n_jobs_parallel = -1  # Use all available cores

# print("\n🔄 Running random trials in parallel...\n")
# Parallel(n_jobs=n_jobs_parallel)(
#     delayed(run_trial)(study, i) for i in range(n_trials_parallel)
# )


# # Randm Exploration
# study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)

# Sequential tuning
study.optimize(objective, n_trials=5, show_progress_bar=True, n_jobs=1)


print("\n✅ Optimization Complete!\n")
print("Best Parameters:", study.best_params)

# Save optimization history to CSV
study_df = study.trials_dataframe()
study_df.sort_values(by="value", ascending=True).to_csv("optuna_optimization_history_weighted_rmse.csv", index=False)


best_model = CustomRandomForestRegressor(**study.best_params)
pipeline = create_pipeline(best_model)

print("\n\n Training model! \n\n")

pipeline.fit(X_train, y_train)
print("\n✅ Model has been trained! \n\n")

joblib.dump(pipeline, 'trained_RF_model_rmse_optimal_2.pkl')
print("\n✅ Trained model saved \n\n")


pred_test = pipeline.predict(X_test)

print("\n Evaluating model! \n\n")

evaluate_model(pred_test, y_test, feature_titles)
