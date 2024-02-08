import pandas as pd
from sklearn import preprocessing
import numpy as np

#read in data
data = pd.read_csv('test1.csv')
#START OF PREPROCESS DATA, Neural Networks perform better with numbers
#preprocess gender so it is 0 for female and 1 for male
label = preprocessing.LabelEncoder()
gender = label.fit_transform(data['Gender'])
data['Gender'] = gender
vAge = label.fit_transform(data['Vehicle_Age'])
data['Vehicle_Age'] = vAge
vDmg = label.fit_transform(data['Vehicle_Damage'])
data['Vehicle_Damage'] = vDmg
#END OF PREPROCESS DATA
#START OF NORMALIZING DATA
data['Age'] = data['Age']/85
data['Region_Code'] = data['Region_Code']/52
data['Vehicle_Age'] = data['Vehicle_Age']/2
data['Policy_Sales_Channel'] = data['Policy_Sales_Channel']/163
data['Vintage'] = data['Vintage']/299
data['Annual_Premium'] = data['Annual_Premium']/47204.2
#END OF NORMALIZING DATA
#CREATE NEURAL NETWORK
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
#split data randomly so 60% is training and 40% is testing
target = data['Annual_Premium']
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=.3, random_state=42)
mlp = MLPRegressor(activation = 'relu', hidden_layer_sizes=(50, 50), alpha= .001, random_state=20)
mlp.fit(X_train, Y_train)
from sklearn.metrics import mean_squared_error
#Make Prediction on x test values
pred = mlp.predict(X_test)
#Calculate accuracy and error metrics
test_set_rsquared = mlp.score(X_test, Y_test)
test_set_rmse = np.sqrt(mean_squared_error(Y_test, pred))
#Print R_squared and RMSE value
print('R_squared value: ', test_set_rsquared)
print('RMSE: ', test_set_rmse)
