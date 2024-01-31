import pandas as pd
from sklearn import preprocessing
data = pd.read_csv('test1.csv', names = ['Gender','Age','DrivingLicense', 'RegionCode', 'PreviouslyInsured', 'VehicleAge', 'VehicleDamage','PolSalesChannel', 'Vintage', 'AnnualPremium'])
#assign first 9 columns to X variable
X = data.iloc[:, 0:9]
#assign data from premium into Y variable
Y = data.select_dtypes(include=[object])
#preprocess all input values to numbers, outputs are already numbers
