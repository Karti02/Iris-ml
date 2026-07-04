from flask import Flask, request, render_template
from datetime import datetime
import pytz
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib') # Loading your trained model
scaler = joblib.load('scaler.joblib') # Loading your scaler
@app.route('/')
def index():
    ct = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
    return render_template('index.html', ct=ct)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    sl = float(request.form.get('sl'))
    sw = float(request.form.get('sw'))
    pl = float(request.form.get('pl'))
    pw = float(request.form.get('pw')) 
    
    features = [[sl, sw, pl, pw]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    pred = prediction[0]
    pred = pred.replace("Iris-", "").lower()
    return render_template(
    "index.html",
    pred=pred,
    ct=datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
)
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    