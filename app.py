from flask import Flask, render_template, request
import pandas as pd
import joblib
# import pickle

app = Flask(__name__)


with open('pipeline.pkl','rb') as f:
    pipeline_saved = joblib.load(f)




@app.route('/')
def index():
    return render_template('./index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    gender = request.form['gender']
    age = request.form['age']
    height = request.form['height']
    weight = request.form['weight']
    bodyTemp = request.form['bodyTemp']
    duration = request.form['duration']
    heartRate= request.form['heartRate']


    data = {
        'Gender': [gender],
        'Age':[age],
        'Height':[height],
        'Weight':[weight],
        'Duration':[duration],
        'Heart_Rate':[heartRate],
        'Body_Temp':[bodyTemp],
    }

    final_data = pd.DataFrame(data, index=[0])

    print(final_data)

    result = pipeline_saved.predict(final_data)[0]
    result = f'{result:.2f}'

    return render_template('index.html', prediction_text='Amount of Calories Burnt: {} Kcal'.format(result))



# commenting for production
# if __name__ == '__main__':
#     app.run()
