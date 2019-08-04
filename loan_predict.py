import pandas as pd  
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import numpy as np
label_enc = preprocessing.LabelEncoder()
data=pd.read_csv('train_loan.csv')
test=pd.read_csv('test_loan.csv')

temp=[]


# print(data['Loan_Amount_Term'].value_counts())

# print(data.isnull().sum())
# print(test.isnull().sum())
data['Gender'].fillna('Male',inplace=True)
test['Gender'].fillna('Male',inplace=True)
data['Dependents'].fillna(0,inplace=True)
test['Dependents'].fillna(0,inplace=True)
data['Married'].fillna('Yes', inplace=True)
test['Married'].fillna('Yes', inplace=True)
data['Self_Employed'].fillna('No', inplace=True)
test['Self_Employed'].fillna('No', inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(360,inplace=True)
test['Loan_Amount_Term'].fillna(360,inplace=True)
data['Credit_History'].fillna(2, inplace=True)
test['Credit_History'].fillna(2, inplace=True)

# print(data.isnull().sum())
# print(test.isnull().sum())

data['eligible']=((data['ApplicantIncome']+data['CoapplicantIncome'])*(data['Loan_Amount_Term']/12))-(data['LoanAmount']*1000)
test['eligible']=((test['ApplicantIncome']+test['CoapplicantIncome'])*(test['Loan_Amount_Term']/12))-(test['LoanAmount']*1000)
for x in data['eligible']:
	if x<0:
		temp.append(0)
	else:
		temp.append(1)

data['eligible']=temp


temp=[]
for x in test['eligible']:
	if x<0:
		temp.append(0)
	else:
		temp.append(1)

test['eligible']=temp


data['Singleton'] = data['Dependents'].map(lambda d: 1 if d=='1' else 0)
data['Small_Family'] = data['Dependents'].map(lambda d: 1 if d=='2' else 0)
data['Large_Family'] = data['Dependents'].map(lambda d: 1 if d=='3+' else 0)
data.drop(['Dependents'], axis=1, inplace=True)

test['Singleton'] = test['Dependents'].map(lambda d: 1 if d=='1' else 0)
test['Small_Family'] = test['Dependents'].map(lambda d: 1 if d=='2' else 0)
test['Large_Family'] = test['Dependents'].map(lambda d: 1 if d=='3+' else 0)
test.drop(['Dependents'], axis=1, inplace=True)


data['eligible']=data['eligible'].astype('int')
test['eligible']=test['eligible'].astype('int')

data['LoanAmount']=data['LoanAmount'].astype('int')
test['LoanAmount']=test['LoanAmount'].astype('int')
data['Loan_Amount_Term']=data['Loan_Amount_Term'].astype('int')
test['Loan_Amount_Term']=test['Loan_Amount_Term'].astype('int')
data['Credit_History']=data['Credit_History'].astype('int')
test['Credit_History']=test['Credit_History'].astype('int')




data['total_income']=data['ApplicantIncome']+data['CoapplicantIncome']
data.drop(['ApplicantIncome','CoapplicantIncome'],axis=1)
data['debt_income_ratio']=data['total_income']/data['LoanAmount']

test['total_income']=test['ApplicantIncome']+test['CoapplicantIncome']
test.drop(['ApplicantIncome','CoapplicantIncome'],axis=1)
test['debt_income_ratio']=test['total_income']/test['LoanAmount']


data['total_income']=data['total_income'].astype('int')
test['total_income']=test['total_income'].astype('int')
data['debt_income_ratio']=data['debt_income_ratio'].astype('int')
test['debt_income_ratio']=test['debt_income_ratio'].astype('int')

# Y=data['Loan_Status']
# data.drop(['Loan_Status','Loan_ID'],axis=1)





for col in test.columns.values:
	if test[col].dtype == 'object':
		data1 = data[col].append(test[col])
		label_enc.fit(data1)
		data[col] = label_enc.transform(data[col])
		test[col] = label_enc.transform(test[col])


Y=data['Loan_Status'].map({'Y':1,'N':0})
data.drop(['Loan_Status','Loan_ID'],axis=1,inplace=True)
test.drop(['Loan_ID'],axis=1,inplace=True)



# data.to_csv("train_temp_loan.csv")


print("Data split")
X_train,X_val,Y_train,Y_val=model_selection.train_test_split(data,Y,test_size=0.20)

print("training")

clf = RandomForestClassifier(
		n_jobs=-1,
		criterion='entropy',
		n_estimators=100,
		max_features=.33,
		max_depth=6,
		min_samples_leaf=4,
		# min_samples_split=3,
		# max_leaf_nodes=35000,
		warm_start=True,
		oob_score=True,
		random_state=321)

clf.fit(X_train,Y_train)

print("RANDOM FOREST SCORE",clf.score(X_val,Y_val))

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,4)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

print(importances)
