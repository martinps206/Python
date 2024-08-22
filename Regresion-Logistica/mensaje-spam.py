import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords

# Cargar datos
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocesamiento
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

nltk.download('stopwords')
stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(stop_words=stop_words)

X = vectorizer.fit_transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de regresión logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Evaluación
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
