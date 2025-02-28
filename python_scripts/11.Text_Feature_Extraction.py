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

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# # Methods - Text Feature Extraction with Bag-of-Words

# %% [markdown]
# In many tasks, like in the classical spam detection, your input data is text.
# Free text with variables length is very far from the fixed length numeric representation that we need to do machine learning with scikit-learn.
# However, there is an easy and effective way to go from text data to a numeric representation using the so-called bag-of-words model, which provides a data structure that is compatible with the machine learning aglorithms in scikit-learn.

# %% [markdown]
# <img src="figures/bag_of_words.svg" width="100%">
#

# %% [markdown]
# Let's assume that each sample in your dataset is represented as one string, which could be just a sentence, an email, or a whole news article or book. To represent the sample, we first split the string into a list of tokens, which correspond to (somewhat normalized) words. A simple way to do this to just split by whitespace, and then lowercase the word. 
#
# Then, we build a vocabulary of all tokens (lowercased words) that appear in our whole dataset. This is usually a very large vocabulary.
# Finally, looking at our single sample, we could show how often each word in the vocabulary appears.
# We represent our string by a vector, where each entry is how often a given word in the vocabulary appears in the string.
#
# As each sample will only contain very few words, most entries will be zero, leading to a very high-dimensional but sparse representation.
#
# The method is called "bag-of-words," as the order of the words is lost entirely.

# %%
X = ["Some say the world will end in fire,",
     "Some say in ice."]

# %%
len(X)

# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X)

# %%
vectorizer.vocabulary_

# %%
X_bag_of_words = vectorizer.transform(X)

# %%
X_bag_of_words.shape

# %%
X_bag_of_words

# %%
X_bag_of_words.toarray()

# %%
vectorizer.get_feature_names()

# %%
vectorizer.inverse_transform(X_bag_of_words)

# %% [markdown]
# # tf-idf Encoding
# A useful transformation that is often applied to the bag-of-word encoding is the so-called term-frequency inverse-document-frequency (tf-idf) scaling, which is a non-linear transformation of the word counts.
#
# The tf-idf encoding rescales words that are common to have less weight:

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X)

# %%
import numpy as np
np.set_printoptions(precision=2)

print(tfidf_vectorizer.transform(X).toarray())

# %% [markdown]
# tf-idfs are a way to represent documents as feature vectors. tf-idfs can be understood as a modification of the raw term frequencies (`tf`); the `tf` is the count of how often a particular word occurs in a given document. The concept behind the tf-idf is to downweight terms proportionally to the number of documents in which they occur. Here, the idea is that terms that occur in many different documents are likely unimportant or don't contain any useful information for Natural Language Processing tasks such as document classification. If you are interested in the mathematical details and equations, see this [external IPython Notebook](http://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/tfidf_scikit-learn.ipynb) that walks you through the computation.

# %% [markdown]
# # Bigrams and N-Grams
#
# In the example illustrated in the figure at the beginning of this notebook, we used the so-called 1-gram (unigram) tokenization: Each token represents a single element with regard to the splittling criterion. 
#
# Entirely discarding word order is not always a good idea, as composite phrases often have specific meaning, and modifiers like "not" can invert the meaning of words.
#
# A simple way to include some word order are n-grams, which don't only look at a single token, but at all pairs of neighborhing tokens. For example, in 2-gram (bigram) tokenization, we would group words together with an overlap of one word; in 3-gram (trigram) splits we would create an overlap two words, and so forth:
#
# - original text: "this is how you get ants"
# - 1-gram: "this", "is", "how", "you", "get", "ants"
# - 2-gram: "this is", "is how", "how you", "you get", "get ants"
# - 3-gram: "this is how", "is how you", "how you get", "you get ants"
#
# Which "n" we choose for "n-gram" tokenization to obtain the optimal performance in our predictive model depends on the learning algorithm, dataset, and task. Or in other words, we have consider "n" in "n-grams" as a tuning parameters, and in later notebooks, we will see how we deal with these.
#
# Now, let's create a bag of words model of bigrams using scikit-learn's `CountVectorizer`:

# %%
# look at sequences of tokens of minimum length 2 and maximum length 2
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_vectorizer.fit(X)

# %%
bigram_vectorizer.get_feature_names()

# %%
bigram_vectorizer.transform(X).toarray()

# %% [markdown]
# Often we want to include unigrams (single tokens) AND bigrams, wich we can do by passing the following tuple as an argument to the `ngram_range` parameter of the `CountVectorizer` function:

# %%
gram_vectorizer = CountVectorizer(ngram_range=(1, 2))
gram_vectorizer.fit(X)

# %%
gram_vectorizer.get_feature_names()

# %%
gram_vectorizer.transform(X).toarray()

# %% [markdown]
# Character n-grams
# =================
#
# Sometimes it is also helpful not only to look at words, but to consider single characters instead.   
# That is particularly useful if we have very noisy data and want to identify the language, or if we want to predict something about a single word.
# We can simply look at characters instead of words by setting ``analyzer="char"``.
# Looking at single characters is usually not very informative, but looking at longer n-grams of characters could be:

# %%
X

# %%
char_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer="char")
char_vectorizer.fit(X)

# %%
print(char_vectorizer.get_feature_names())

# %% [markdown]
# <div class="alert alert-success">
#     <b>EXERCISE</b>:
#      <ul>
#       <li>
#       Compute the bigrams from "zen of python" as given below (or by ``import this``), and find the most common trigram.
# We want to treat each line as a separate document. You can achieve this by splitting the string by newlines (``\n``).
# Compute the Tf-idf encoding of the data. Which words have the highest tf-idf score? Why?
# What changes if you use ``TfidfVectorizer(norm="none")``?
#       </li>
#     </ul>
# </div>

# %%
zen = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""

# %% {"deletable": true, "editable": true}
# # %load solutions/11_ngrams.py
