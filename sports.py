from flask import Flask, request
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import random
import os

# Initialize flask app
app = Flask(__name__)

acc = 0

# On the add page, take two arguments, add them, and return the string
@app.route("/add", methods=['GET'])
def second_api():
    # array mapping numbers to flower names
    classes = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]
    
    # get values for each component, return error message if not a float
    try:
        values = [[float(request.args.get(component)) for component in ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]]]
    except TypeError:
        return "An error occured\nUsage: 127.0.0.1:5000?length=NUM&diameter=NUM&height=NUM&whole_weight=NUM&shucked_weight=NUM&viscera_weight=NUM&shell_weight=NUM"
    
    # Otherwise, return the prediction.
    prediction = find_data(values)
    rtn_str = "{:d} years (i.e. {:d} rings) with an accuracy of {:f}%".format(int(prediction), int(prediction), acc)
    return rtn_str

def find_data(values):
	# Importing data from a csv file
	dataset = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/abalone_csv.csv')

	# Adding an Id tag to the dataframe
	dataset['Id'] = range(4177)

	# Ways to visualize the data:
	rows, cols = dataset.shape


	# Check how many of each species we have
	dataset.groupby('Class_number_of_rings').size()

	# splitting up the labels and the values for each species:
	feature_columns = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
	X = dataset[feature_columns].values
	Y = dataset['Class_number_of_rings'].values


	# Encoding Labels (Turning string species names into integers)
	# setosa -> 0
	# versicolor -> 1
	# virginica -> 2
	le = LabelEncoder()
	Y = le.fit_transform(Y)
	# Splitting into training and test datasets:
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)

	# Creating the learning model
	knn_classifier = KNeighborsClassifier(n_neighbors=10)

	# Fitting the model with the training data
	knn_classifier.fit(X_train, Y_train)
	Y_pred = knn_classifier.predict(X_test)
	print(Y_pred)

	# Finding Accuracy:
	accuracy = accuracy_score(Y_test, Y_pred)*100
	print('Accuracy of model: ' + str(round(accuracy, 2)) + ' %.')
	global acc
	acc = round(accuracy, 2)
	cm = confusion_matrix(Y_test, Y_pred)
	return knn_classifier.predict(values)[0]

# Run the application
app.run()


