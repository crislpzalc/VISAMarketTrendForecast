import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
import joblib



"""Download the VISA Stock Data 2024 from Kaggle"""
# Download latest version
path = kagglehub.dataset_download("umerhaddii/visa-stock-data-2024")

print("Path to dataset files:", path)

"""Find the dataset"""
# Find the csv file in the path downloaded before
for file in os.listdir(path):
  if file.endswith(".csv"):
    csv_file = os.path.join(path, file)
    break

# Convert the dataset in DataFrame
df = pd.read_csv(csv_file)



"""Looking at the Data"""
# show the first five rows
df.head()

# show basic information
df.info()

# show statistical info about each column
df.describe()

df.hist(bins=50, figsize=(8,8), color="#D29BFD")



"""Create Test Set"""
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

"""Set aside the Test Set and explore the data of the Train Set"""
# copy of the train set to explore it
train_copy = train_set.copy()

# correlations between the attributes and the volume
corr_matrix = train_copy.corr(numeric_only=True)
corr_matrix["Volume"].sort_values(ascending=False)

# correlations between the attributes and the close price
corr_matrix["Close"].sort_values(ascending=False)

# create some new variables and eliminate others
train_copy["High-Low"] = train_copy["High"] - train_copy["Low"]
train_copy["Close-Open"] = train_copy["Close"] - train_copy["Open"]
train_copy = train_copy.drop(columns=["Low", "High", "Adj Close", "Open", "Date"])

# we have to do the same with the test set
test_copy = test_set.copy()
test_copy["High-Low"] = test_copy["High"] - test_copy["Low"]
test_copy["Close-Open"] = test_copy["Close"] - test_copy["Open"]
test_copy = test_copy.drop(columns=["Low", "High", "Adj Close", "Open", "Date"])

# correlations between the attributes and the volume
corr_matrix = train_copy.corr(numeric_only=True)
corr_matrix["Volume"].sort_values(ascending=False)

# binary variable
train_copy["Trend"] = (train_copy["Close"].shift(-1) > train_copy["Close"]).astype(int)

# we have to do the same for test_set
test_copy["Trend"] = (test_copy["Close"].shift(-1) > test_copy["Close"]).astype(int)

# correlations between the attributes and the trend
corr_matrix = train_copy.corr(numeric_only=True)
corr_matrix["Trend"].sort_values(ascending=False)

# drop useless variables
train_copy = train_copy.drop(columns=["Close-Open"])



"""Prepare the data"""
y_train = train_copy["Trend"].copy()
train_copy = train_copy.drop("Trend", axis=1)

# we do the same for test set
y_test = test_copy["Trend"].copy()
test_copy = test_copy.drop("Trend", axis=1)

"""Create a pipeline to preprocess the data"""
# to add missing data
imputer = SimpleImputer(strategy="median")

# to transform the data
log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")

# see if there is a bias
plt.figure(figsize=(4,3))
plt.hist(train_copy['Volume'], bins=50, color="#D29BFD")
plt.title("Distribution of Volume")
plt.show()

# calculate the skewness for all the numeric columns
skewness = train_copy.skew(numeric_only=True)
print(skewness)

"""Logaritmic transform in the skewnessed variables. We use a pipeline to simplify the process."""

preprocessing = ColumnTransformer([("log_transform", Pipeline([
    ("imputer", imputer), ("log", log_transformer)]), ["Volume", "High-Low"]),
                                   ("imputer_only", imputer, ["Close"])
                                   ], remainder='passthrough')

train_copy_prepared = preprocessing.fit_transform(train_copy)
train_copy_prepared.shape

preprocessing.get_feature_names_out()



"""Select the model"""
# Starting with Linear Regression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(train_copy, y_train)

trend_predictions = lin_reg.predict(train_copy)
print('trend_predictions:', trend_predictions[:5].round())
print('actual values:', y_train.iloc[:5].values)

# Decision Tree Regression.
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(train_copy, y_train)

trend_predictions = tree_reg.predict(train_copy)
print('trend_predictions:', trend_predictions[:5])
print('actual values:', y_train.iloc[:5].values)

tree_rmse = mean_squared_error(y_train, trend_predictions, squared=False)
print(tree_rmse)

"""Cross validation"""
tree_rmses = -cross_val_score(tree_reg, train_copy, y_train,
                              scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(tree_rmses).describe())

# Random Forest Regressor
rforest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
rforest_reg.fit(train_copy, y_train)

rforest_rmses = -cross_val_score(rforest_reg, train_copy, y_train,
                              scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(rforest_rmses).describe())

"""Grid Search"""
full_pipeline = Pipeline([("preprocessing", preprocessing),
 ("random_forest", RandomForestRegressor(random_state=42))])

param_grid = {"random_forest__n_estimators" : [50, 100, 150, 200],
              "random_forest__max_depth": [None, 10, 20, 30],
              "random_forest__min_samples_split": [2, 5, 10],
              "random_forest__min_samples_leaf": [1, 2, 4]}

grid_search = GridSearchCV(full_pipeline, param_grid, cv=10,
                           scoring='neg_root_mean_squared_error', n_jobs=-1)

grid_search.fit(train_copy, y_train)

grid_search.best_params_

grid_search.best_estimator_

-grid_search.best_score_



"""Evaluate the Model with the Test Set"""
final_predictions = grid_search.best_estimator_.predict(test_copy)

final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
print(final_rmse)

# confidence interval
confidence = 0.99
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))



"""Model persistance"""
joblib.dump(grid_search.best_estimator_, "VISA_market_trend_forecast.pkl")
