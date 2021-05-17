import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random as r
import numpy as np
from scipy import stats

# Reading the creditcard.csv file using pandas
data_input = pd.read_csv("creditcard.csv")

print("\n", data_input.describe())

# Data quality check

print("\nChecking for NULL/MISSING values...")
print(round(100 * (data_input.isnull().sum() / len(data_input)), 2).sort_values(ascending=False))
print("\nPercentage of missing values in each row...")
print(round(100 * (data_input.isnull().sum(axis=1) / len(data_input)), 2).sort_values(ascending=False))
print("\nDuplicate check in the data-set...")
data_input_duplicate = data_input.copy()
data_input_duplicate.drop_duplicates(subset=None, inplace=True)
print("\nOriginal data-set shape: ", data_input.shape)
print("\nDuplicated data-set dropped attribute shape: ", data_input_duplicate.shape)
data_input = data_input_duplicate
del data_input_duplicate


# Exploratory Data Analysis

def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
        ax.set_title(feature + " Distribution", color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()


# Plotting histograms with outliers
draw_histograms(data_input, data_input.columns, 8, 4)
print("\nClass counts...")
print("\n", data_input.Class.value_counts())

# Removing the outliers
data_input = data_input[(np.abs(stats.zscore(data_input.iloc[:, 1:29])) < 3).all(axis=1)]

# Plotting histograms without outliers
draw_histograms(data_input, data_input.columns, 8, 4)

# Separating out the dependent and independent variables
feature_columns = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14",
                   "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
    , "Amount"]
X = data_input[feature_columns]
y = data_input["Class"]

# Splitting the data-set into 20% test set and 80% training set
print("\nSplitting the data-set into 20% test set and 80% training set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data set using StandardScaler()
print("\nScaling the data set using StandardScaler()...")
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Building the classifier
print("\nBuilding the classifier...")
classifier = LogisticRegression()

# Training the Logistic Regression classifier
print("\nTraining the Logistic Regression classifier...")
classifier.fit(X_train_std, y_train)

# Predict the response for test dataset
print("\nPredict the response for test dataset...")
y_pred = classifier.predict(X_test_std)

# Printing the classification factors i.e. Accuracy and Confusion Matrix
print("\nAccuracy: ", metrics.accuracy_score(y_test, y_pred) * 100, "%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix...")
print(cm)

# Making prediction for a test data
test_data = [
    [20000, r.random() * 10, r.random() * 10, r.random() * 10, r.random() * 10, r.random() * 10, r.random() * 10,
     r.random() * 10, r.random() * 10, r.random() * 100, r.random() * 100, r.random() * 100, r.random() * 100,
     r.random() * 100, r.random() * 100, r.random() * 100, r.random() * 100, r.random() * 100, r.random() * 10,
     r.random() * 10, r.random() * 10, r.random() * 10, r.random() * 10, r.random() * 10, r.random() * 10,
     r.random() * 10,
     r.random() * 10, r.random() * 10, r.random() * 10, 200]]
predicted_class = classifier.predict(test_data)
print("\nTest data: ", test_data)
print("\nPredicted class: ", predicted_class[0])

# Generating the heat-map based upon the confusion-matrix data
print("\nGenerating heat map...")
sns.heatmap(cm, annot=True)
plt.show()
print("\nHeat map generated...")
