# CoachTaekwondo AI

## Présentation

CoachTaekwondo AI est un projet de vision par ordinateur et de deep learning visant à reconnaître automatiquement deux techniques de taekwondo : **Ap Tchagui** et **Yop Tchagui**.

Le projet s'appuie sur MediaPipe pour l'extraction des points anatomiques du corps et sur un réseau de neurones pour la classification des techniques. Une application Streamlit permet ensuite d'utiliser le modèle de manière interactive.

## Objectifs

* Détecter automatiquement une technique de taekwondo à partir d'une image.
* Étudier l'apport de différentes méthodes de préparation des données.
* Comparer plusieurs approches de réduction et de sélection de variables.
* Déployer le modèle dans une application utilisable par un pratiquant ou un entraîneur.

## Technologies utilisées

* Python
* MediaPipe
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit

## Structure du projet

```text
CoachTaekwondo/
│
├── 01_augmentation_dataset.ipynb
├── 02_extraction_coordonnees.ipynb
├── 03_analyse_exploratoire_coordonnees.ipynb
├── 04_ACP.ipynb
├── 05_selection_variables.ipynb
├── 06_Réseau de neuronnes.ipynb
│
├── dataset/          # Images du dataset
├── exports/          # Données extraites et fichiers CSV
├── images/           # Illustrations et visualisations
├── models/           # Modèles entraînés et artefacts
├── appli/            # Application Streamlit
└── doc/              # Documentation du projet
```

## Pipeline du projet

1. Constitution et augmentation du dataset.
2. Extraction des coordonnées des landmarks MediaPipe.
3. Analyse exploratoire des données.
4. Réduction de dimension (ACP).
5. Sélection de variables.
6. Entraînement et évaluation des modèles.
7. Déploiement dans une application Streamlit.

## Résultats

Le modèle retenu est un réseau de neurones multicouche entraîné sur l'ensemble des coordonnées sélectionnées. Il permet d'obtenir une précision supérieure à 90 % sur le jeu de test.

## Application

L'application Streamlit est disponible dans le dossier `appli/`.

Consulter le README du dossier `appli` pour les instructions d'installation et d'exécution.
