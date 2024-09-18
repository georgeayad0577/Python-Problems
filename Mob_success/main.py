# import libraries
import datasets
import pandas as pd
import matplotlib.pyplot as plt
#for importing linear
from sklearn import linear_model
from sklearn import metrics
#for importing decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# load dataset
dt = pd.read_csv('Mob_Success.csv')
print(dt.describe())

# define x and y
#x = dt.drop(columns=['Rating'])
#y = dt['Rating']
x = dt.data
y = dt.target
print(x.shape)
print(y.shape)

# load dataset + create x,y matrices
x, y = datasets.load_data(return_x_y=True)
print(x.shape)
print(y.shape)

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# data dimension
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# linear regression model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# prediction results
# print model performance
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)
print("Mean Squared error: ", metrics.mean_squared_error(y_test, y_pred))
Print("accuracy: ", accurance_score(y_test, y_pred))

print(dt.feature_names)
# make scatter
plt.scatterplot(y_test, y_pred)
plt.plot(x, y_pred, color='red')
plt.show()

# trying to make Decision Tree Classifier

#Model = DecisionTreeClassifier()
#model.fit(x_train, y_train)
#Prediction = model.predict(x_test)

#score = accuracy_score(y_test, Prediction)
#print(score)

#tree.export_graphviz(model, out_file='dt-recommender.dot', feature_name=['Reviews', 'Rating'], class_names=sorted(y.unique()), label='all', rounded=True, filled=True)
