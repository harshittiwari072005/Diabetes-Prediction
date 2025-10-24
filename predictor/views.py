from django.shortcuts import render
import joblib
import numpy as np

# load your trained model
model = joblib.load('predictor/diabetes_prediction_model.pkl')

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        data = [
            float(request.POST['Pregnancies']),
            float(request.POST['Glucose']),
            float(request.POST['BloodPressure']),
            float(request.POST['SkinThickness']),
            float(request.POST['Insulin']),
            float(request.POST['BMI']),
            float(request.POST['DiabetesPedigreeFunction']),
            float(request.POST['Age'])
        ]
        prediction = model.predict([np.array(data)])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return render(request, 'home.html', {'result': result})
    return render(request, 'home.html')

