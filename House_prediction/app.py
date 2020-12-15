from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd
app = Flask('__name__')
model=joblib.load('HistGradientBoostingRegressor.pkl')
pipline=joblib.load(open('pipline.pkl','rb'))
@app.route('/')
def home():
    return render_template('form.html')
@app.route('/predict',methods=["POST"])
def predict():
 longitude = request.form.get('longitude')
 latitude = request.form.get('latitude')
 housing_median_age = request.form.get('housing_median_age')
 total_rooms = request.form.get('total_rooms')
 total_bedrooms = request.form.get('total_bedrooms')
 population = request.form.get('population')
 households = request.form.get('households')
 median_income = request.form.get('median_income')
 ocean_proximity = request.form.get('ocean_proximity')
 rooms_per_household = float(total_rooms)/ float(households)
 bedrooms_per_room = float(total_bedrooms)/ float(total_rooms)
 population_per_household = float(population)/ float(households)
 arr = np.array([longitude, latitude, housing_median_age, total_rooms, population, households,
 median_income,ocean_proximity, rooms_per_household,bedrooms_per_room, population_per_household])
 dataF=pd.DataFrame(data=[arr], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'population', 'households', 'median_income','ocean_proximity', 'rooms_per_household',
       'bedrooms_per_room', 'population_per_household'])
 new_features=pipline.transform(dataF)
 prediction=model.predict(new_features)
 return render_template('form.html',prediction_text='{}$'.format(prediction))
if (__name__=='__main__'):
    app.run(debug=True)

