# Analyse des Sentiments des Commentaires de Films

## Description

Ce projet a pour but d'analyser les sentiments des commentaires de films en utilisant un modèle de régression logistique. Les sentiments sont classés en deux catégories : **Bien** (positif) et **Mal** (négatif). Le jeu de données utilisé contient des critiques de films annotées avec des sentiments, permettant ainsi de construire et d'évaluer un modèle de classification binaire.

## Objectifs

1. **Comprendre les sentiments des spectateurs** : En analysant les commentaires des films, nous pouvons obtenir des informations précieuses sur la perception du public vis-à-vis des films.
2. **Automatiser la classification des sentiments** : Utiliser des techniques de traitement du langage naturel (NLP) et de machine learning pour automatiser l'identification des sentiments dans les critiques de films.
3. **Créer un modèle prédictif** : Construire un modèle de régression logistique capable de prédire si un commentaire est positif ou négatif.

## Jeu de Données

Le jeu de données utilisé dans ce projet est le [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) disponible sur Kaggle. Il contient 50 000 critiques de films avec des annotations de sentiments. Nous voulions beaucoup de données pour mieux prédire, ce jeu est assez conséquent mais du coup il est en anglais. Nous prédisons donc les commentaires anglais.

## Étapes du Projet

1. **Chargement des Données** : Importation du jeu de données et affichage des premières lignes pour avoir un aperçu.
2. **Préparation des Données** : Séparation des commentaires et des étiquettes, et conversion des sentiments en valeurs binaires.
3. **Division des Données** : Séparation du jeu de données en ensembles d'entraînement et de test pour évaluer le modèle.
4. **Vectorisation TF-IDF** : Conversion des commentaires en vecteurs TF-IDF pour une représentation numérique appropriée.
5. **Entraînement du Modèle** : Entraînement d'un modèle de régression logistique sur les données d'entraînement.
6. **Évaluation du Modèle** : Évaluation des performances du modèle sur l'ensemble de test en utilisant des métriques d'exactitude et un rapport de classification.
7. **Prédictions** : Utilisation du modèle entraîné pour prédire les sentiments de nouveaux commentaires.

## Pourquoi un modèle supervisé ?

Choisir un modèle supervisé pour ce projet permet d'utiliser les étiquettes annotées des sentiments pour entraîner un modèle précis et fiable, capable de généraliser efficacement à de nouveaux commentaires. Cela optimise la performance de prédiction en exploitant les relations explicites entre les caractéristiques des données et les étiquettes de sortie. De plus, les données sont faciles à étiquetter, il donnera des prédictions plus fiables.

## Utilisation

Pour exécuter ce projet, suivez les étapes ci-dessous :

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/JustBen71/ReviewAi.git
   pip install pandas scikit-learn nltk
## Auteurs

Benjamin RANDAZZO, Mathilde VOLLET, Capucine MADOULAUD, Lou TIGROUDJA