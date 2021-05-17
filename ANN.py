import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import random as r
import numpy as np
from scipy import stats

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


draw_histograms(data_input, data_input.columns, 8, 4)
print("\nClass counts...")
print("\n", data_input.Class.value_counts())

# Removing the outliers
data_input = data_input[(np.abs(stats.zscore(data_input.iloc[:, 1:29])) < 3).all(axis=1)]

# Separating out the dependent and independent variables
feature_columns = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14",
                   "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
                    , "Amount"]
X = data_input[feature_columns]
y = data_input.Class

# Splitting the data-set into 10% test set and 90% training set
print("\nSplitting the data-set into 10% test set and 90% training set...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Scaling the data set using StandardScaler()
print("\nScaling the data set using StandardScaler()...")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Adding the input and first hidden layer
# For adding the layers we use the Dense library of keras
# input_dim is the number of independent variables and output_dim is the ceil of half of the total no of variables
# init stands for weight initialization
print("\nCreating all the input, hidden and output layers...")
classifier = Sequential()
classifier.add(Dense(16, kernel_initializer=keras.initializers.random_uniform, activation="relu", input_dim=30))

# Adding the second hidden layer
# input_dim not required as the input will be from the first layer
classifier.add(Dense(16, kernel_initializer=keras.initializers.random_uniform, activation="relu"))

# Adding the output layer
classifier.add(Dense(1, kernel_initializer=keras.initializers.random_uniform, activation="sigmoid"))

classifier.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training the classifier
# batch_size is basically the no of datas to be trained in a single epoch
# In all there are 150 epochs
print("\nTraining the ANN classifier...")
classifier.fit(X_train, y_train, batch_size=100, epochs=150)

# Predicting the test set results
print("\nPredicting the test set results...")
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Printing the classification factors i.e. Accuracy and Confusion Matrix
print("\nAccuracy: ", metrics.accuracy_score(y_test, y_pred)*100, "%")

# Printing the confusion matrix
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print("\n", cm)

# Making prediction for a test data
test_data = [[20000, r.random()*10, r.random()*10, r.random()*10, r.random()*10, r.random()*10, r.random()*10,
              r.random()*10, r.random()*10, r.random()*100, r.random()*100, r.random()*100, r.random()*100,
              r.random()*100, r.random()*100, r.random()*100, r.random()*100, r.random()*100, r.random()*10,
              r.random()*10, r.random()*10, r.random()*10, r.random()*10, r.random()*10, r.random()*10, r.random()*10,
              r.random()*10, r.random()*10, r.random()*10, 200]]
predicted_class = classifier.predict(test_data)
print("\nTest data: ", test_data)
print("\nPredicted class: ", predicted_class[0])

# Generating the heat-map based upon the confusion-matrix data
print("\nGenerating heat map...")
sns.heatmap(cm, annot=True)
plt.show()
print("\nHeat map generated...")
