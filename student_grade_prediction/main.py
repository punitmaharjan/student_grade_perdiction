from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df=pd.read_csv('student-mat.csv')
df.isnull().sum()
df.dropna(inplace=True)

features = df[['G1', 'G2', 'Medu', 'failures']].values
target = df['G3'].values

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=1)

def normalize(features_train, features_test):
    mean = np.mean(features_train, axis=0)
    std = np.std(features_train, axis=0)
    features_train_normalized = (features_train - mean) / std
    features_test_normalized = (features_test - mean) / std
    return features_train_normalized, features_test_normalized
features_train_norm, features_test_norm = normalize(features_train, features_test)

features_train_norm = np.hstack((np.ones((features_train_norm.shape[0], 1)), features_train_norm))
features_test_norm = np.hstack((np.ones((features_test_norm.shape[0], 1)), features_test_norm))

def cost(features, target, theta):
    num_of_samples = len(target) 
    predictions = features.dot(theta)
    cost = 1 / 2 * num_of_samples * np.sum((predictions - target) ** 2)
    return cost

def gradient_descent(features, target, theta, alpha, num_of_iterations):
    num_of_samples = len(target)
    cost_history = np.zeros(num_of_iterations)

    for i in range(num_of_iterations):
        predictions = features.dot(theta)
        loss = predictions - target
        theta = theta - (alpha * (features.T.dot(loss))) / num_of_samples
        cost_history[i] = cost(features, target, theta)

    return theta, cost_history

def predict_new_data(new_data, features_train, theta):
    i = 0
    for col in range(features_train.shape[1]):
        col_mean = np.mean(features_train[:, col])
        col_std = np.std(features_train[:, col])
        new_data[i] = (new_data[i] - col_mean) / col_std
        i += 1
    new_data = np.insert(new_data, 0, 1)
    target_predict = new_data.dot(theta)
    return target_predict

# hyperparameters of gradient descent function

alpha = 0.01  #learning rate
num_of_iterations = 1000

#theta -> vector of parameters (weights) that will be learned by the model during training.
theta = np.zeros(features_train_norm.shape[1])
theta, cost_history = gradient_descent(features_train_norm, target_train, theta, alpha, num_of_iterations)

target_predictions = features_test_norm.dot(theta)

app = Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['GET', 'POST'])
def home():
    result = ''
    if (request.method == 'POST'):
        #pickled_model = pickle.load(open('./student_grade_prediction_v3.1.pkl', 'rb'))
        
        #getting input values from the user
        medu = float(request.form['medu'])
        failures = float(request.form['failures'])
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])

        # making predictions using pickled model
        data = np.array([G1, G2, medu, failures])
        #features_train = pickled_model['features_train']
        #theta = pickled_model["theta"]

        result = predict_new_data(data, features_train, theta)
        result = round(result, 2)

        return render_template('predict.html', result = result)
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run()