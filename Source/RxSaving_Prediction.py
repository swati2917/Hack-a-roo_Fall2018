# RxSaving Python


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Creating the Panda DataFrame from File
sdu_data = pd.read_csv('SDU_2017.csv', dtype={'SuppressionUsed': np.bool, 'UtilizationType': np.object,
                                              'State': np.object, 'NDC': np.object})

# Removing the row with State = 'XX' and SuppressionUse == as don't have any value for Drug

sdu_data_filter = sdu_data[(sdu_data['State'] != 'XX') & (sdu_data['SuppressionUsed'] == False & (sdu_data['State'] == 'KS'))]

features_cols =['LabelerCode', 'ProductCode', 'PackageSize', 'Year', 'Quarter', 'NDC', 'NonMedicaidAmountReimbursed',
                'UnitsReimbursed', 'NumberofPrescriptions']

# sdu_data_filter.info()

X = sdu_data_filter[features_cols]
y = sdu_data_filter.MedicaidAmountReimbursed

print(X.head())
print(y.head())


# Prediction Using Keras Library

# Training and Test Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fitting the linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set Result

y_pred = regressor.predict(X_test)


y_compare = np.vstack((y_test, y_pred)).T

print(y_compare[:5, :])

from sklearn import metrics

# print the accuracy score of predicted and actual values on test set

print('\n\nAccuracy of the Linear Regression for KS State', metrics.r2_score(y_test, y_pred))

# Plot the Test Prediction , Test Value on Graph

#plt.scatter(y_test, y_test, color='yellow')
plt.plot(y_test, y_pred, color='yellow', linewidth=2)
plt.xlabel('Drug vs Number Prescription/UnitReimbursed')
plt.ylabel(' Medicaid Drug Price ')
plt.title('Medicaid Drug Price Prediction For KS State')

plt.show()






































