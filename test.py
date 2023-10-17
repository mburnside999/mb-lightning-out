import pandas
from sklearn import linear_model

df = pandas.read_csv("cars.csv")

columns = ['Weight', 'Volume', 'CO2']
df2 = pandas.read_csv('cars.csv', usecols=columns)
print(df2.corr())


X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)


#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])
print(regr.coef_)
print(regr.feature_names_in_)
print("prediction "+str(predictedCO2))