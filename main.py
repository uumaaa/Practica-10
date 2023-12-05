from Models import naive_bayes
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
model = naive_bayes.NaiveBayesClassifier()

# Fijamos una semilla para reproducibilidad
np.random.seed(42)

# Generar datos de entrenamiento
X = np.random.randn(200, 3)  # 100 ejemplos, 3 características
y = np.random.choice([0, 1], size=200)  # Clases binarias (0 o 1)



model.fit(X_train,y_train)
model.predict(X_test)
