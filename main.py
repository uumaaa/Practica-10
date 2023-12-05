from Models import naive_bayes
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Fijamos una semilla para reproducibilidad
np.random.seed(42)
X = np.random.randn(200, 3) 
y = np.random.choice([0], size=200)
model = naive_bayes.NaiveBayesClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=40)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(y_predict)
print(y_test)
accuracy = accuracy_score(y_test,y_predict)
print(accuracy)