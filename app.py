from flask import Flask, render_template, request, url_for, redirect
import joblib
from sklearn.datasets import load_iris

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    sepal_length = request.form.get('sepal_length')
    sepal_width = request.form.get('sepal_width')
    sepal_length = request.form.get('petal_length')
    sepal_width = request.form.get('petal_width')
    
    m = [sepal_length, sepal_width, sepal_length, sepal_width]

    measures = [float(measure) for measure in m]
    classifier = joblib.load('trained_model.pkl')
    prediction = classifier.predict([measures])

    iris = load_iris()
    # Possible output -> 'setosa' (0), 'versicolor' (1) or 'virginica' (2)
    kind_of_iris = iris['target_names'][prediction][0]

    return render_template('prediction.html',kind_of_iris=kind_of_iris)

if __name__ == '__main__':
    app.run(debug=True)
