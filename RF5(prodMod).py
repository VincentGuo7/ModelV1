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
## Runs and trains a new Model



######### Getting Feature titles of the data ################
def get_titles(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)

    print(f"Columns in dataset: {df.columns.tolist()}")

    non_feature_cols = ['date', 'lat', 'lon']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    return feature_cols




###### Splitting Data #######
def train_val_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42):
    # assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must add up to 1."

    # X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # val_ratio = val_size / (train_size + val_size)
    
    # X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_state)
    
    # print(f"\nSize of training dataset: {len(y_train)}\n")

    # return X_train, X_val, X_test, y_train, y_val, y_test


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
    

    

####### Creating RFR with customised hyperparameters ########
class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, 
                 n_estimators=210,
                 max_features='sqrt',
                 max_samples=0.7,
                 max_depth=16,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_impurity_decrease=0.05
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





# ############### Feature Importance via MDI ##################

# def plot_feature_importance(pipeline, X_shape, base_feature_names, save_path=None):
#     # Extract trained RandomForest model
#     rf_model = pipeline.named_steps['regressor'].model_

#     if not hasattr(rf_model, "feature_importances_"):
#         raise ValueError("Model does not have feature_importances_ attribute")

#     importances = rf_model.feature_importances_

#     n_timesteps = X_shape[1]
#     feature_names = [
#         f"{fname}_t{t+1}"
#         for t in range(n_timesteps)
#         for fname in base_feature_names
#     ]

#     # Create dataframe of importances
#     feat_imp_df = pd.DataFrame({
#         "feature": feature_names,
#         "importance": importances
#     }).sort_values(by="importance", ascending=False)

#     # Plot per-time-step importances (your current plot)
#     plt.figure(figsize=(12, 8))
#     plt.barh(feat_imp_df["feature"], feat_imp_df["importance"])
#     plt.gca().invert_yaxis()
#     plt.xlabel("Feature Importance (MDI)")
#     plt.title("Random Forest Feature Importance (Per Timestep)")
#     plt.tight_layout()
#     plt.show()

#     # ---------- NEW: Aggregate importance per base feature ----------
#     # Strip off "_tX" from each feature name
#     feat_imp_df["base_feature"] = feat_imp_df["feature"].str.extract(r"^(.*)_t\d+$")

#     agg_df = feat_imp_df.groupby("base_feature", as_index=False)["importance"].sum()
#     agg_df = agg_df.sort_values(by="importance", ascending=False)

#     # Plot aggregated importances
#     plt.figure(figsize=(10, 6))
#     plt.barh(agg_df["base_feature"], agg_df["importance"])
#     plt.gca().invert_yaxis()
#     plt.xlabel("Total Importance (Summed Over Time)")
#     plt.title("Random Forest Feature Importance (Aggregated)")
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path)
#         print(f"Feature importance plot saved to: {save_path}")

#     plt.show()

#     return feat_imp_df, agg_df




############## ACTION CODE #################

# feature_titles = get_titles('climax_processed_NA.parquet')
X = np.load("X_data_final.npy")
y_full = np.load("y_data_final.npy")  # full features including lat/lon

print("\n✅ Data has been loaded into Arrays! \n\n")

X_train, X_test, y_train_full, y_test = train_val_test_split(X,y_full)

y_train = y_train_full[:, 5:]


model = CustomRandomForestRegressor()
pipeline = create_pipeline(model)

print("\n Training model! \n\n")

pipeline.fit(X_train, y_train)
print("\n✅ Model has been trained! \n\n")

joblib.dump(pipeline, 'trained_RF_model_3.pkl', compress=3)
print("\n✅ Trained model saved \n\n")


# feat_imp_df, agg_df = plot_feature_importance(pipeline, X_train.shape, feature_titles, save_path='feature_importance.png')


# pred_test = pipeline.predict(X_test)
# # pred_test = pipeline.predict(X_test)


# print("\n Evaluating model! \n\n")

# evaluate_model(pred_test, y_test, feature_titles)
