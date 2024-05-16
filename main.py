import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Charger le jeu de données
df = pd.read_csv('dataset.csv')

# Afficher un échantillon du jeu de données
print(df.head())

# Séparer les données et les étiquettes
X = df['Phrase']
y = df['Sentiment']

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un pipeline avec la vectorisation TF-IDF, la normalisation et le modèle de régression logistique
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', StandardScaler(with_mean=False)),
    ('logreg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500))
])

# Entrainer le modèle
pipeline.fit(X_train, y_train)

# Prédire les intentions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Evaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

new_comments = ["This film was really incredible", "I did not like this film", "This film was average"]
predictions = pipeline.predict(new_comments)
intention_map = {0: 'negative', 1: 'rather negative', 2: 'neutral', 3: 'rather positive', 4: 'positive'}

for comment, pred in zip(new_comments, predictions):
    print(f'Comment: {comment}')
    print(f'Sentiment: {intention_map[pred]}')
    print()