# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% {"deletable": true, "editable": true}
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown] {"deletable": true, "editable": true}
# # Pipelining estimators

# %% [markdown] {"deletable": true, "editable": true}
# In this section we study how different estimators maybe be chained.

# %% [markdown] {"deletable": true, "editable": true}
# ## A simple example: feature extraction and selection before an estimator

# %% [markdown] {"deletable": true, "editable": true}
# ### Feature extraction: vectorizer

# %% [markdown] {"deletable": true, "editable": true}
# For some types of data, for instance text data, a feature extraction step must be applied to convert it to numerical features.
# To illustrate we load the SMS spam dataset we used earlier.

# %% {"deletable": true, "editable": true}
import os

with open(os.path.join("datasets", "smsspam", "SMSSpamCollection")) as f:
    lines = [line.strip().split("\t") for line in f.readlines()]
text = [x[1] for x in lines]
y = [x[0] == "ham" for x in lines]

# %% {"deletable": true, "editable": true}
from sklearn.model_selection import train_test_split

text_train, text_test, y_train, y_test = train_test_split(text, y)

# %% [markdown] {"deletable": true, "editable": true}
# Previously, we applied the feature extraction manually, like so:

# %% {"deletable": true, "editable": true}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)

clf.score(X_test, y_test)

# %% [markdown] {"deletable": true, "editable": true}
# The situation where we learn a transformation and then apply it to the test data is very common in machine learning.
# Therefore scikit-learn has a shortcut for this, called pipelines:

# %% {"deletable": true, "editable": true}
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
pipeline.fit(text_train, y_train)
pipeline.score(text_test, y_test)

# %% [markdown] {"deletable": true, "editable": true}
# As you can see, this makes the code much shorter and easier to handle. Behind the scenes, exactly the same as above is happening. When calling fit on the pipeline, it will call fit on each step in turn.
#
# After the first step is fit, it will use the ``transform`` method of the first step to create a new representation.
# This will then be fed to the ``fit`` of the next step, and so on.
# Finally, on the last step, only ``fit`` is called.
#
# ![pipeline](figures/pipeline.svg)
#
# If we call ``score``, only ``transform`` will be called on each step - this could be the test set after all! Then, on the last step, ``score`` is called with the new representation. The same goes for ``predict``.

# %% [markdown] {"deletable": true, "editable": true}
# Building pipelines not only simplifies the code, it is also important for model selection.
# Say we want to grid-search C to tune our Logistic Regression above.
#
# Let's say we do it like this:

# %% {"deletable": true, "editable": true}
# This illustrates a common mistake. Don't use this code!
from sklearn.model_selection import GridSearchCV

vectorizer = TfidfVectorizer()
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

clf = LogisticRegression()
grid = GridSearchCV(clf, param_grid={'C': [.1, 1, 10, 100]}, cv=5)
grid.fit(X_train, y_train)

# %% [markdown] {"deletable": true, "editable": true}
# ### What did we do wrong?

# %% [markdown] {"deletable": true, "editable": true}
# Here, we did grid-search with cross-validation on ``X_train``. However, when applying ``TfidfVectorizer``, it saw all of the ``X_train``,
# not only the training folds! So it could use knowledge of the frequency of the words in the test-folds. This is called "contamination" of the test set, and leads to too optimistic estimates of generalization performance, or badly selected parameters.
# We can fix this with the pipeline, though:

# %% {"deletable": true, "editable": true}
from sklearn.model_selection import GridSearchCV

pipeline = make_pipeline(TfidfVectorizer(), 
                         LogisticRegression())

grid = GridSearchCV(pipeline,
                    param_grid={'logisticregression__C': [.1, 1, 10, 100]}, cv=5)

grid.fit(text_train, y_train)
grid.score(text_test, y_test)

# %% [markdown] {"deletable": true, "editable": true}
# Note that we need to tell the pipeline where at which step we wanted to set the parameter ``C``.
# We can do this using the special ``__`` syntax. The name before the ``__`` is simply the name of the class, the part after ``__`` is the parameter we want to set with grid-search.

# %% [markdown] {"deletable": true, "editable": true}
# <img src="figures/pipeline_cross_validation.svg" width="50%">

# %% [markdown] {"deletable": true, "editable": true}
# Another benefit of using pipelines is that we can now also search over parameters of the feature extraction with ``GridSearchCV``:

# %% {"deletable": true, "editable": true}
from sklearn.model_selection import GridSearchCV

pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

params = {'logisticregression__C': [.1, 1, 10, 100],
          "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (2, 2)]}
grid = GridSearchCV(pipeline, param_grid=params, cv=5)
grid.fit(text_train, y_train)
print(grid.best_params_)
grid.score(text_test, y_test)

# %% [markdown] {"deletable": true, "editable": true}
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Create a pipeline out of a StandardScaler and Ridge regression and apply it to the Boston housing dataset (load using ``sklearn.datasets.load_boston``). Try adding the ``sklearn.preprocessing.PolynomialFeatures`` transformer as a second preprocessing step, and grid-search the degree of the polynomials (try 1, 2 and 3).
#       </li>
#     </ul>
# </div>

# %% {"deletable": true, "editable": true}
# # %load solutions/15A_ridge_grid.py
