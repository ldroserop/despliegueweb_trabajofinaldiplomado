from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Cargar el modelo y el preprocesador
with open('svm_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Valor inicial de la predicción
initial_prediction = 0

# Rutas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener los datos del formulario
        college = request.form['college']
        personal_ai_use = int(request.form['personal_ai_use'])
        school_ai_use = int(request.form['school_ai_use'])
        ai_interest = int(request.form['ai_interest'])

        # Convertir el nombre del colegio en un código numérico
        college_codes = {'Selecciona una carrera': 0, 'Arts & Media': 1, 'Science, Engineering, & Technology': 2, 'Education': 3, 'Nursing & Health Care': 4, 'Humanities & Social Sciences': 5, 'Business': 6, 'Theology': 7}
        college_code = college_codes[college]

        # Realizar la predicción
        input_data = pd.DataFrame([[college_code, personal_ai_use, school_ai_use, ai_interest]],
                                  columns=['What college are you in?', 
                                           'On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for personal use?', 
                                           'On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for school-related tasks?', 
                                           'On a scale from 1 to 5, how interested are you in pursuing a career in Artificial Intelligence?'])
        prediction = pipeline.predict_proba(input_data)[0][1] * 100  # Convertir la probabilidad a porcentaje
    else:
        prediction = 0

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
