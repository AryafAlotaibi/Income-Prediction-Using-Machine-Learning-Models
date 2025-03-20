
import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation and analysis
import matplotlib.pyplot as plt  # for data visualization
from sklearn.model_selection import train_test_split  # for data splitting
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # for feature scaling
from sklearn.svm import SVC  # for Support Vector Classifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report  # for evaluation
import seaborn as sns # for statistical graphics
from sklearn.tree import DecisionTreeClassifier #to build the decision tree model
from sklearn.tree import plot_tree #for visualizing decision tree models
from sklearn.neighbors import KNeighborsClassifier #estimation of the Knn model and outcome report

"""------------------------------------------------------"""

# @title 2- Importing the selected dataset and visualizing the dataset contents
# Define column names
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# Load the training dataset
train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
train_data = pd.read_csv(train_url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)

# Load the testing dataset
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
test_data = pd.read_csv(test_url,
                        header=None, # No header row in the file, as column names are not provided in the file
                        names=column_names, # Assign the custom column names to the DataFrame from the `column_names` list
                        na_values=" ?",  # "?" in the dataset are missing values
                        skipinitialspace=True,  # Ignore any extra spaces after delimiters (commas) when parsing the file
                        skiprows=1  # Skip the first row in the file (metadata)
                        )

# Check the results
print("The number of train data:")
print(train_data.shape[0])
print("\nThe number of test data:")
print(test_data.shape[0])
print("\n____________________________________________")

test_data.head()

"""------------------------------------------------------"""

# @title 3a- Train Data
# Count to check if there are null values in the column
train_data.isna().sum()

train_data

# replace ? value to null value
train_data.replace('?', None, inplace=True)
test_data.replace('?', None, inplace=True)

# count to check if there are null values in the column
train_data.isna().sum()

# Display a summary of the DataFrame
train_data.info()

# Count the occurrences of each class (<=50K, >50K) in the 'income' column
train_data.income.value_counts()

# Visualize income distribution for training data
sns.countplot(x='income', data=train_data)
plt.show()

# @title 3b- Test Data
test_data.isna().sum() # count the null values in the column

test_data

test_data['income'] = test_data['income'].replace('<=50K.', '<=50K')
test_data['income'] = test_data['income'].replace('>50K.', '>50K')

# Visualize income distribution for test data
sns.countplot(x='income', data=test_data)
plt.show()

# Visualize age distribution
plt.figure(figsize=(8, 6))
sns.histplot(train_data['age'], bins=20, kde=True, color='coral')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

sns.countplot(x='income', hue='sex', data=train_data)
plt.show()

"""------------------------------------------------------

In the following step we combine training and test data to perform preprocessing, which ensures consistency and prevents "data leakage" between the training and testing stages
"""

# @title 4- Data preprocessing
#Combine the Training and Test Data
combined_data = pd.concat([train_data, test_data], ignore_index=True)
combined_data.head(4)

combined_data.shape[0]

combined_data.isna().sum() # count to check if there are null values in the column

"""the results show three class with null "missing" values, to fix this we fill the missing values with the **Most Frequent Value**"""

# Fill missing values in 'workclass'
mode_value_workclass = combined_data['workclass'].mode()[0]
combined_data['workclass'] = combined_data['workclass'].fillna(mode_value_workclass)

# Fill missing values in 'occupation'
mode_value_occupation = combined_data['occupation'].mode()[0]
combined_data['occupation'] = combined_data['occupation'].fillna(mode_value_occupation)

# Fill missing values in 'native-country'
mode_value_native_country = combined_data['native_country'].mode()[0]
combined_data['native_country'] = combined_data['native_country'].fillna(mode_value_native_country)

combined_data.isna().sum() # check if there are still null values in the column

combined_data

"""------------------------------------------------------

Encoding converts non-numeric categorical data (e.g., 'Male', 'Female') into a numerical format that the model can understand while preserving the information about the categories.
Here, **One-Hot Encoding** is used for nominal data to avoid introducing artificial ordering.
"""

#One Hot Encoding Categorical Variables
#get all categorical columns
categ_columns = combined_data.select_dtypes(['object']).columns

#convert all categorical columns to numeric
combined_data[categ_columns] = combined_data[categ_columns].apply(lambda x: pd.factorize(x)[0])

#print head of data after convert
combined_data

# Visualize income distribution
sns.countplot(x='income', data=combined_data)
plt.show()

"""Apply Normalization since SVM relies on distance to define margins, and k-NN calculates distances to identify neighbors making them sensitive to feature scales.

"""

# Separate features and target
X_combined_data = combined_data.drop('income', axis=1)
y_combined_data = combined_data['income']

# Apply Min-Max Normalization to features
scaler = MinMaxScaler()
X_combined_data_normalized = scaler.fit_transform(X_combined_data)


# Convert back to DataFrame for better readability
combined_data= pd.DataFrame(X_combined_data_normalized, columns=X_combined_data.columns)

combined_data =combined_data.join(y_combined_data)

combined_data

"""Separate the Combined Data Back Into Train and Test"""

# Get the original number of rows for the training data
train_size = len(train_data)

# Separate the combined data back into train and test sets
train_data_separated = combined_data.iloc[:train_size]
test_data_separated = combined_data.iloc[train_size:]

# Check the results
print("Train Data Separated:")
print(train_data_separated.shape[0])
print("\nTest Data Separated:")
print(test_data_separated.shape[0])

# Separate features and target
X_train = train_data_separated.drop('income', axis=1)
y_train = train_data_separated['income']
X_test = test_data_separated.drop('income', axis=1)
y_test = test_data_separated['income']

test_data.head()

"""------------------------------------------------------"""

# @title 5- Setting Up the SVM Model

# Initialize the SVM classifier with default parameters
SVM_classifier = SVC()

# initialize SVM classifier
SVM_classifier = SVC(kernel='rbf')

"""**Training the SVM Model**"""

# X_train represent the training features, and y_train represent the training labels.
# The initialized SVM classifier is 'SVM_classifier'

# Train the SVM classifier using the training data
SVM_classifier.fit(X_train, y_train)

"""**Testing and Evaluating the SVM Model**"""

# Make predictions using the test data
y_pred_svm = SVM_classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", accuracy)

# Make a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_svm)

# Plot confusion matrix
plt.figure(figsize=(5,3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Evaluating the model
print(confusion_matrix(y_test, y_pred_svm))  # confusion matrix
print("____________________________________________\n")
print(classification_report(y_test, y_pred_svm))  # classification report

"""Based on the provided classification report:
the model shows very high precision and recall for class 0, indicating the instances of that class are reliably set. The recall for class 1 is appreciably worse, hinting toward undetected instances. Although the accuracy is very good at 84%, further investigation of the balance between precision and recall or possible strategies to improve the performance.

------------------------------------------------------
"""

# @title 6- Setting Up the Decision Tree Model
# perform training with entropy
# Decision tree with entropy
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 42,max_depth = 4, min_samples_leaf =5)
# Fit the model
clf.fit(X_train, y_train)

"""**Training the Decision Tree Model**"""

# Predictions on the test set
y_pred = clf.predict(X_test)

# Confusion matrix and performance metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

"""**Testing and Evaluating the Decision Tree Model**"""

# Create a heatmap for the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', linewidths=0.5, linecolor='black')

# Add labels and title
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Display the plot
plt.show()

# Plot the decision tree
plt.figure(figsize=(15, 7))  # Adjust the size as needed
plot_tree(clf, feature_names=X_train.columns, class_names=['income_<=50K', 'income_>50K'], filled=True, rounded=True)

# Add title
plt.title('Decision Tree Visualization')

# Display the plot
plt.show()

"""Based on the provided classification report: the model shows the overall accuracy is strong, with  high accuracy for class 0 but struggles with classifying 1 (many False Negatives)

------------------------------------------------------
"""

# @title 7- Setting Up the KNN Model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
# Predictions on the test set
y_pred_knn = knn.predict(X_test)

# Confusion matrix and performance metrics
cm = confusion_matrix(y_test, y_pred_knn)
accuracy = accuracy_score(y_test, y_pred_knn)
report = classification_report(y_test, y_pred_knn)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_knn)
print("Accuracy:", accuracy)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_knn)

# Plot confusion matrix
plt.figure(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

"""Based on the provided classification report: the model shows that the overall accuracy is strong, but the imbalance in performance across classes suggests room for improvement, especially for the minority class (Class 1)."""