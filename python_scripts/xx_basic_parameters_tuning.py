# %% [markdown]
# # Introduction to scikit-learn: basic model hyper-parameters tuning
#
# In this lecture note, we aim at:
# * illustrate the influence of changing model parameters;
# * illustrate how to tune these hyper-parameters;
# * evaluate the model performance together with hyper-parameters tuning.
#
# ## Recall of basic preprocessing and model fitting
#
# In the previous lecture note, we show how to preprocessed different type of
# data and integrate this preprocessing in a machine learning pipeline
# containing a predictor.
#
# We will recall this example. First, we will load the data and organize it
# into a `data` and a `target` variable. The ultimate goal is to train a
# predictor able to estimate the wages from different censing data.

# %%
import os
import pandas as pd

df = pd.read_csv(os.path.join('datasets', 'cps_85_wages.csv'))
target_name = "WAGE"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

# %% [markdown]
# Once the dataset is loaded, we will split it into a training and testing sets

# %%
from sklearn.model_selection import train_test_split

df_train, df_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
# Once the data split, we can define our preprocessing to transform differently
# the numerical and categorical data

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

binary_encoding_columns = ['MARR', 'SEX', 'SOUTH', 'UNION']
one_hot_encoding_columns = ['OCCUPATION', 'SECTOR', 'RACE']
scaling_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']

preprocessor = ColumnTransformer([
    ('binary-encoder', OrdinalEncoder(), binary_encoding_columns),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'),
     one_hot_encoding_columns),
    ('standard-scaler', StandardScaler(), scaling_columns)
])

# %% [markdown]
# After defining the preprocessing, we will use a linear regressor (i.e. ridge)
# to predict wages.

# %%
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, Ridge())
model.fit(df_train, target_train)
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f}"
)

# %% [markdown]
# ## The issue of having the best model parameters
#
# When using the `Ridge` regressor, one could notice that we are using the
# default parameters by omitting setting explicitly these parameters.
#
# For such regressor, the parameter `alpha` is governing the penalty; in other
# words, how much our model should "trust" (or fit) the training data.
#
# Therefore, the default value of `alpha` is never certified to give the best
# model.
#
# We can make a quick experiment by changing the value of `alpha` and see the
# impact of this parameter on the model performance.

# %%
alpha = 1
model = make_pipeline(preprocessor, Ridge(alpha=alpha))
model.fit(df_train, target_train)
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f} with alpha={alpha}"
)

alpha = 10000
model = make_pipeline(preprocessor, Ridge(alpha=alpha))
model.fit(df_train, target_train)
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(df_test, target_test):.2f}  with alpha={alpha}"
)

# %% [markdown]
# ## Finding the best model hyper-parameters via exhaustive parameters search
#
# We see that the parameter `alpha` as a significative impact on the model
# performance and that finding the best value for this parameter is crucial.
# However, this parameter should be tuned with cross-validation such that
# we find a parameter. In short, we will set the parameter, train our model
# on some data, and evaluate the performance of our model on some left out
# data. Ideally, we will select the parameter leading to the optimal
# performance on the testing set. Scikit-learn provides a `GridSearchCV`
# estimator which will handle the cross-validation for us.

# %%
from sklearn.model_selection import GridSearchCV

model = make_pipeline(preprocessor, Ridge())

# %% [markdown]
# We will see that we need to provide the name of the parameter to be set.
# Thus, we can use the method `get_params()` to have the list of the parameters
# of the model which can set during the grid-search

# %%
print("The model hyper-parameters are:")
print(model.get_params())

# %% [markdown]
# The parameter `'ridge__alpha'` is the parameter for which we would like
# different values. Let see how to use the `GridSearchCV` estimator for doing
# such search.

# %%
import numpy as np

param_grid = {'ridge__alpha': np.linspace(0.001, 1000, num=20)}
model_grid_search = GridSearchCV(model, param_grid=param_grid)
model_grid_search.fit(df_train, target_train)
print(
    f"The R2 score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)

# %% [markdown]
# The `GridSearchCV` estimator takes a `param_grid` parameter which defines
# all possible parameters combination. Once the grid-search fitted, it can be
# used as any other predictor by calling `predict` and `predict_proba`.
# Internally, it will use the model with the best parameters found during
# `fit`. You can now about these parameters by looking at the attribute
# `best_params_`

# %%
print(f"The best set of parameters is: {model_grid_search.best_params_:.1f}")

# %% [markdown]
# The parameters during the grid-search need to be specificy. Instead, one
# could randomly generate (following a specific distribution) the parameter
# candidates. The `RandomSearchCV` allows for such stochastic search. It can
# be used similarly to the `GridSearchCV` but one has to specified the
# distributions instead of the parameter values.

# %%
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'ridge__alpha': uniform(loc=50, scale=100)}
model_grid_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=20
)
model_grid_search.fit(df_train, target_train)
print(
    f"The R2 score using a {model_grid_search.__class__.__name__} is "
    f"{model_grid_search.score(df_test, target_test):.2f}"
)
print(f"The best set of parameters is: {model_grid_search.best_params_:.1f}")

# %% [markdown]
# ## Notes on search efficiency
#
# Be aware that sometimes, scikit-learn provides some `EstimatorCV` classes
# which will perform internally the cross-validation in such way that it will
# more computationally efficient. We can give the example of the `RidgeCV`
# which can be used to find the best `alpha` in a more efficient way than what
# we previously did with the `GridSearchCV`.

# %%
import time
from sklearn.linear_model import RidgeCV

# define the different alphas to try out
param_grid = {"alpha": (0.1, 1.0, 10.0)}

model = make_pipeline(preprocessor, RidgeCV(alphas=param_grid['alpha']))
start = time.time()
model.fit(df_train, target_train)
print(f"Time elapsed to train RidgeCV: {time.time() - start:.3f} seconds")

model = make_pipeline(
    preprocessor, GridSearchCV(Ridge(), param_grid=param_grid)
)
start = time.time()
model.fit(df_train, target_train)
print(f"Time elapsed to make a grid-search on Ridge: {time.time() - start:.3f} "
      f"seconds")

# %% [markdown]
# ## Combining evaluation and hyper-parameters search
#
# We saw that we are using a cross-validation for searching the best model
# parameters. In addition, we previously saw that we can use cross-validation
# to evaluate the model performance. If we would like to combine both aspects,
# one needs to perform "nested" cross-validation. The "outer" cross-validation
# will be applied to assess the model while the "inner" cross-validation will
# set the hyper-parameters of the model on the data set provided by the "outer"
# cross-validation. In practice, it is equivalent of including, `GridSearchCV`,
# `RandomSearchCV`, or any `EstimatorCV` in a `cross_val_score` or
# `cross_validate` function call.

# %%
from sklearn.model_selection import cross_val_score

model = make_pipeline(preprocessor, RidgeCV())
score = cross_val_score(model, data, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(f"The different scores obtained are: \n{score}")

# %% [markdown]
# Be aware that such training might involve a variation of the hyper-parameters
# of the model. When analyzing such model, you should not only look at the
# overall model performance but look at the hyper-parameters variations as
# well.
