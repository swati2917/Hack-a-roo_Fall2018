# RxSaving Python


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


sdu_data = pd.read_csv('SDU_2017.csv', dtype={'SuppressionUsed': np.bool, 'UtilizationType': np.object,
                                              'State': np.object, 'NDC': np.object})

dndc_data = pd.read_csv('Disease_NDC.csv', dtype={'NDC': np.str})


# Removing the row with State = 'XX' as it won't give us correct picture related to prediction


sdu_data_filter = sdu_data[(sdu_data['State'] != 'XX') & (sdu_data['SuppressionUsed'] == False)]

# sdu_data_filter.info()

X = sdu_data_filter.iloc[:, 0:20]
y = sdu_data_filter.iloc[:, 12]

# Data Processing Steps

# Missing Values
from sklearn.preprocessing import  Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X.values[:, 9:14])
X.values[:, 9:14] = imputer.transform(X.values[:, 9:14])


# Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])


# Data Analysis

# Top 10 Costiliest Rx Drug Paid by Gorvernment

drug = X.groupby('NDC')['MedicaidAmountReimbursed'].sum().reset_index()

drug_sort = drug.sort_values('MedicaidAmountReimbursed', ascending=False).head(10)


# Logic to get product Name
drug_sort_pname = []


for i, r in drug_sort.iterrows():
    for j, k in X.iterrows():
        if r['NDC'] == k['NDC']:
            drug_sort_pname.append(k['ProductName'])
            break


drug_sort['PNAME'] = drug_sort_pname
drug_sort["NDC"] = drug_sort["NDC"].astype(object)
drug_sort["PNAME"] = drug_sort["PNAME"].astype(str)




# Plot Top 10 Rx Drugs in US
plt.figure(1, figsize=(12, 12))
sns.barplot(drug_sort["PNAME"], drug_sort["MedicaidAmountReimbursed"], linewidth=1, edgecolor="k"*len(drug_sort))
plt.title("Top 10 Costliest Rx Drugs in US ", color='b')


# Top 10 State by Spending Medicaid Rx Drug

drug_Rx_state = X.groupby('State')['MedicaidAmountReimbursed'].sum().reset_index()

drug_Rx_state_sort = drug_Rx_state.sort_values('MedicaidAmountReimbursed', ascending=False).head(10)


# Plot Top 10 State by Spending Medicaid Rx Drug
plt.figure(2, figsize=(13, 7))

ax = plt.scatter("State", "MedicaidAmountReimbursed", data=drug_Rx_state_sort,
                 c=drug_Rx_state_sort["MedicaidAmountReimbursed"], cmap="inferno", s=900, alpha=.7,
                 linewidth=2, edgecolor="k",)
plt.title('Top 10 State by Spending Medicaid Rx Drug', color='b')
plt.xlabel("State")
plt.ylabel("Total MedicaidAmount(Multiple of 100 Million)")


# Top 10 State With Highest Number of Rx Prescription

presc_state = X.groupby('State')['NumberofPrescriptions'].sum().reset_index()

presc_state_sort = presc_state.sort_values('NumberofPrescriptions', ascending=False).head(10)

# Plot Top 10 State With Highest Number of Prescription

plt.figure(3, figsize=(10, 8))

ax = sns.barplot(y=presc_state_sort['NumberofPrescriptions'], x=presc_state_sort['State'], palette="plasma",
                 linewidth=1, edgecolor="k"*15)
plt.xlabel("States")
plt.ylabel("Number of Prescription")
plt.title("State With Highest Number of Prescription", color='b')


# Costliest Diseases by Medicaid Spending


diseases = X.groupby('NDC')['MedicaidAmountReimbursed'].sum().reset_index()

diseases_sort = diseases.sort_values('MedicaidAmountReimbursed', ascending=False).head(10)

diseases_sort_dname = []


for i, r in diseases_sort.iterrows():
    for j, k in dndc_data.iterrows():
        if r['NDC'] == k['NDC']:
            diseases_sort_dname.append(k['MList'])
            break

diseases_sort['DNAME'] = diseases_sort_dname
diseases_sort["NDC"] = diseases_sort["NDC"].astype(object)
diseases_sort["DNAME"] = diseases_sort["DNAME"].astype(str)

# Plot Costliest Disease
plt.figure(4, figsize=(10, 8))

ax = sns.barplot(y=diseases_sort['MedicaidAmountReimbursed'], x=diseases_sort['DNAME'], palette="jet_r", alpha=.8,
                 linewidth=2, edgecolor="k"*len(diseases_sort))

plt.title("Costliest Diseases by Medicaid Spending")
plt.xlabel("Diseases")
plt.ylabel("Spending in Multiple 100 Millions")



# Top Costliest Diseases in California


sdu_data_filter_1 = sdu_data[(sdu_data['State'] == 'CA') & (sdu_data['SuppressionUsed'] == False)]

diseases_state = sdu_data_filter_1.groupby('NDC')['MedicaidAmountReimbursed'].sum().reset_index()

diseases__state_sort = diseases_state.sort_values('MedicaidAmountReimbursed', ascending=False).head(10)

diseases_state_sort_dname = []


for i, r in diseases__state_sort.iterrows():
    for j, k in dndc_data.iterrows():
        if r['NDC'] == k['NDC']:
            diseases_state_sort_dname.append(k['MList'])
            break

#diseases_sort['DNAME'] = diseases_state_sort_dname
#diseases_sort["NDC"] = diseases_sort["NDC"].astype(object)
#diseases_sort["DNAME"] = diseases_sort["DNAME"].astype(str)
print(diseases_state_sort_dname)
print(diseases__state_sort)

plt.show()





































