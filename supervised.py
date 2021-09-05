# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# %%
def load_bank():
    df=pd.read_csv('data/bank-full.csv', sep=';',header=0)
    # Basic data cleanup to get rid of bad values and turn Y into 0-1 val
    df=df.dropna(axis=1, how='all')
    df=df.dropna(axis=0, how='any')
    cat_cols=[]
    i = 0 
    for eachcol in df.dtypes:
        if eachcol.name=="object":
            cat_cols.append(df.columns[i])
        i=i+1
    # Convert the string values into integers, and give each value its own column, hot encode
    df=pd.get_dummies(df,columns=cat_cols)
    df.head()
    X=df.iloc[:,0:-2]
    y=df['y_yes']
    # Use sklearn to split up the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Could Add a RandomState number to set the rand kernel
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_normalized=scaler.transform(x_train)
    x_test_normalized=scaler.transform(x_test)
    return x_train_normalized, x_test_normalized, y_train, y_test
x_train, x_test, y_train, y_test = load_bank()
# %% [markdown]
# # Decision Tree 
# %%
clf = DecisionTreeClassifier(max_depth=5)
model=clf.fit(x_train,y_train)
accuracy = model.score(x_test, y_test)
print("Decision Tree accuracy: {0:.3f}%".format(accuracy))

# %% [markdown]
# # Neural Network

# %%
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

model = clf.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("Neural Net Accuracy: {0:.3f}%".format(accuracy))

# %% [markdown]
# # Boosting

# %%
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
model = clf.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print("Boosting Accuracy: {0:.3f}%".format(accuracy))

# %% [markdown]
# # Support Vector Machines

# %%
clf = svm.SVC()
model = clf.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("SVM Accuracy: {0:.3f}%".format(accuracy))

# %% [markdown]
# # k-Nearest Neighbors

# %%
clf = KNeighborsClassifier(3)
model = clf.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("k-NN Accuracy: {0:.3f}%".format(accuracy))



