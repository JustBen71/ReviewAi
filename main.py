import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# Convertir les commentaires en vecteurs TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Prédire les intentions sur l'ensemble de test
y_pred = model.predict(X_test_tfidf)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Prédire l'intention pour de nouveaux commentaires
new_comments = ["This film was really incredible", "I did not like this film", "This film was average"]
new_comments_tfidf = vectorizer.transform(new_comments)
predictions = model.predict(new_comments_tfidf)
for comment, pred in zip(new_comments, predictions):
    if pred == 0 or pred == 1 :
        print(f'Commentaire: {comment} - Intention: negative')
    elif pred == 2 :
        print(f'Commentaire: {comment} - Intention: neutral')
    else :
        print(f'Commentaire: {comment} - Intention: positive')