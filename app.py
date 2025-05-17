from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('models/rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))


def predict(rf_classifier, scaler, footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature):
    features = np.array([[footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature]])
    scaled_features = scaler.transform(features)
    result = rf_classifier.predict(scaled_features)
    return result[0]
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/pred",methods=['GET','POST'])
def pred():
    if request.method == 'POST':
        footfall = int(request.form['footfall'])
        tempMode = int(request.form['tempMode'])
        AQ = int(request.form['AQ'])
        USS = int(request.form['USS'])
        CS = int(request.form['CS'])
        VOC = int(request.form['VOC'])
        RP = int(request.form['RP'])
        IP = int(request.form['IP'])
        Temperature = int(request.form['Temperature'])

        prediction = predict(model, scaler, footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature)
        prediction_test = "The Machine has Failure" if prediction == 1 else "The Machine has no Failure"

        return render_template('index.html', prediction=prediction_test)



if __name__ == '__main__':
    app.run(debug=True)