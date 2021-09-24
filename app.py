import pickle
from flask import Flask, render_template,request
import numpy as np

app = Flask(__name__)
MyModel = pickle.load(open('car prediction model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    model = request.form['model']
    km = request.form['km']
    hp = request.form['hp']
    gearingtype = request.form['gearing type']
    bodytype = request.form['body type']

    #prediction = MyModel.predict([['Model', 'KilloMiter','body color']])[0]
    #X = df[['km', 'hp', 'model dummy', 'body type dummy', 'gearing type dummy']]
    prediction = MyModel.predict([[km, hp, model, bodytype, gearingtype]])[0]

    return render_template('index.html', output="$" + str(np.round(prediction, 2)))

if __name__ == '__main__':
    app.run(debug=True)