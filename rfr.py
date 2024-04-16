import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Cargar los datos
data = pd.read_csv('dataset.csv')

# Convertir variables categóricas a códigos si es necesario
data['What college are you in?'] = data['What college are you in?'].astype('category').cat.codes

# Escoger las columnas relevantes como características y la columna objetivo
features = data[['What college are you in?', 
                 'On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for personal use?', 
                 'On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for school-related tasks?', 
                 'On a scale from 1 to 5, how interested are you in pursuing a career in Artificial Intelligence?']]
target = data['Do you know what Chat-GPT is?']  # Asumiendo que 'Do you know what Chat-GPT is?' es binaria

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.6, random_state=42)

# Crear un pipeline con preprocesamiento y SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True))
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo
predictions = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Guardar el pipeline completo (modelo y preprocesador)
with open('svm_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
