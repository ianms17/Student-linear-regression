import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# read csv file into a pandas dataframe
data = pd.read_csv("student-mat.csv", sep=";")

# resize the data frame to only include these 6 column labels
data = data[['G1', 'G2', 'G3', 'studytime', 'absences', 'failures']]

# predictor refers to the label of the attribute we will be trying to predict
predictor = "G3"

# x gets us all the labels, y gets us all the data
X = np.array(data.drop([predictor], 1))
Y = np.array(data[predictor])

# create a train/test model where 10% of the data is reserved for testing, returns data into arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# create a linear regression model and fit a line with x_train and y_train data
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

# display the model accuracy, coefficients, and the y-intecept
print("Accuracy: ", accuracy)
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# create the predictions model and iterate through all the test data points, display predicted grade
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# create a chart double bar graph to show the prediction vs the actual
fig, ax = plt.subplots()
width = 0.5
x_pos = np.arange(len(predictions))

# create the bars, bar1=predicted scores, bar2=actual scores
bar1 = ax.bar(x_pos, predictions, width=width)
bar2 = ax.bar(x_pos + width, y_test, width=width)

ax.set_title("Comparison of Predicted and Actual Scores")
ax.legend((bar1[0], bar2[0]), ('Prediction', 'Actual'))
ax.autoscale_view()
plt.show()
