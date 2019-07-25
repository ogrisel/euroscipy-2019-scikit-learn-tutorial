---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%matplotlib inline
```

```python
import seaborn as sns
sns.set_context('talk')
```

```python
# from matplotlib.axes._axes import _log as matplotlib_axes_logger
# matplotlib_axes_logger.setLevel('ERROR')
```

# Interpret scikit-learn machine learning models


## Dataset: Current Population adult_census (1985)


We will use data from the "Current Population adult_census" from 1985 and fetch it from [OpenML](http://openml.org/).

```python
import pandas as pd

adult_census = pd.read_csv('datasets/cps_85_wages.csv')
```


We can get more information regarding by looking at the description of the dataset.

```python
# TODO show description of the datasets
# iframe from https://www.openml.org/d/534 or text file in the repo
```

The data are stored in a pandas dataframe.

```python
adult_census.head()
```

```python title="target_names = ['WAGE']"

```

The column **WAGE** is our target variable (i.e., the variable which we want
to predict). You can note that the dataset contains both numerical and
categorical data.


First, let's get some insights by looking at the **marginal** links between
the different variables. Only *numerical* variables will be used.

```python
sns.pairplot(adult_census, diag_kind='kde');
```

We can add some additional information to the previous plots.

```python
g = sns.PairGrid(adult_census)
g = g.map_upper(
    sns.regplot, scatter_kws={"color": "black", "alpha": 0.2, "s": 30},
    line_kws={"color": "red"}
)
g = g.map_lower(sns.kdeplot, cmap="Reds_d")
g = g.map_diag(sns.kdeplot, shade=True, color='r')
```

We can already have some intuitions regarding our dataset:

* The "WAGE" distribution has a long tail and we could work by taking the
`log` of the wage; * For all 3 variables, "EDUCATION", "EXPERIENCE", and
"AGE", the "WAGE" is increasing when these variables are increasing; * The
"EXPERIENCE" and "AGE" are correlated.
