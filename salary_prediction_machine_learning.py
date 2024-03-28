# -*- coding: utf-8 -*-
"""Salary Prediction

  --------------------------- FOR BETTER AND CLEAR GO TO Salary_Prediction_Machine_learning.ipyb ----------------------------

# Library :- Pandas, Numpy, Sickit learn, matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading csv file and removing empty and NAN value

dataset = pd.read_csv('salary_data.csv')
dataset = dataset.dropna() # drop the rows containg empty value and NAN values
dataset

"""# predecting salary based on experience"""

dataset = dataset[['Years of Experience', 'Salary']]
dataset

"""# extract or seperate expereince and salary"""

dataset.iloc[: , 0:-1]

x = dataset.iloc[: , 0:-1].values
x

y = dataset.iloc[ : , 1: ].values
y

"""# spliting dataset into train and test"""

from sklearn.model_selection import train_test_split as split

x_train, x_test, y_train, y_test = split(x,y, test_size= 0.1, random_state=1)

print(x_train)

"""# check the length of the train"""

print("Total Length :", len(x))
print("Length of train : " , len(x_train))
print("Length of test : " , len(x_test))

"""# train model"""

from sklearn.linear_model import LinearRegression as lr

model = lr()   # instance linearRegression
model.fit(x_train, y_train)

"""# **Testing model  and predict salary**"""

def pr_salary():
  exp = float(input("Enter years of experience : "))

  exp = np.array([[exp]])

  print(exp)


  # predict salary

  predict_salary = model.predict(exp)

  print(predict_salary)

pr_salary()

pr_salary()

pr_salary()

pr_salary()

"""
# #            
# #
# #
# ***evaluated***
# #
# #
# #"""

dataset = dataset[dataset['Years of Experience'] == 6]
dataset

x_test

y_test

"""# x test to list"""

#x_test.tolist()

predict_salary = model.predict(x_test)
predict_salary

# for testing data

predict_salary = model.predict(x_test)

data_list = []

for i in range(len(x_test)):
  data = [x_test.tolist()[i][0], predict_salary[i][0], y_test.tolist()[i][0]]
  data_list.append(data)

df = pd.DataFrame(data_list, columns = ['Years of Experience', 'Predicted Salary', 'Actual Salary'])
df.sort_values(by = "Years of Experience")

"""# Graph"""

plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, model.predict(x_train), color = "blue")
plt.show()

plt.scatter(x_test, y_test, color = "blue")
plt.plot(x_train, model.predict(x_train) , color = "red")
plt.show()

plt.scatter(y_test,model.predict(x_test),color = 'red')

z = [i for i in range(100)]
plt.scatter(z,z,color = 'red')

