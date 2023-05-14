# Loan Prediction Model

This repository contains code for a loan prediction model using Support Vector Machine (SVM) algorithm. The model is built using Python and various libraries such as NumPy, pandas, seaborn, and scikit-learn.

## Data Collection

The loan dataset is loaded into a pandas DataFrame.

```python
# Importing the Dependencies
import pandas as pd

# Loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('/content/dataset.csv')
```

## Data Preprocessing

We perform various preprocessing steps to get the data ready for training our SVM model.

```python
# Dropping missing values
loan_dataset = loan_dataset.dropna()

# Label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}}, inplace=True)
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# Separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']
```

## Data Visualization

Visualizations are created to gain insights from the loan dataset. The graphs provide information about the relationship between different variables and the loan status.

```python
# Education & Loan Status
import seaborn as sns
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

# Marital Status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
```

## Model Training

A Support Vector Machine (SVM) model is trained using the preprocessed data.

```python
# Importing the Dependencies
from sklearn.model_selection import train_test_split
from sklearn import svm

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Support Vector Machine Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

## Model Evaluation

The accuracy of the trained model is evaluated using both the training data and the test data.

```python
# Importing the Dependencies
from sklearn.metrics import accuracy_score

# Training data accuracy
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

# Test data accuracy
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data:', test_data_accuracy)
```

## Making Predictions

The trained model can be used to make predictions on new data.

Feel free to explore the code in this repository and adapt it for your own loan prediction tasks.

## Requirements

The necessary dependencies for running the code are listed in the `requirements.txt` file. You can install them using the following command:

```
pip install -r requirements.txt
```

Make sure you have Python installed on your system before running the code.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
