# ============================================================
# Projet      : CoachTaekwondo
# Fichier     : model_utils.py
# Auteur      : Aurélie RIVIERE
# Version     : 1.0.0
# Date        : Juin 2026
#
# Description :
# Module regroupant les fonctions liées à l'intelligence
# artificielle et au traitement d'image :
# - chargement des modèles,
# - initialisation de MediaPipe,
# - extraction des landmarks,
# - prédiction du réseau de neurones.
#
# Ce module est utilisé par app.py.
# ============================================================

# Imports des bibliothèques
import os
import streamlit as st
import numpy as np
import joblib
import cv2
import mediapipe as mp
import tensorflow as tf

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from landmarks_config import LANDMARK_NAMES


#---------------------------------------
# Chargement des modèles 
#---------------------------------------

# Importation de la liste des landmarks conservés depuis le fichier de configuration utilisé pendant l'entraînement
from landmarks_config import LANDMARK_NAMES

# Répertoire dans lequel se trouve ce fichier
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Charge les fichiers nécessaires à la prédiction
@st.cache_resource
def load_models():
    """
    Le décorateur @st.cache_resource permet de charger
    les ressources une seule fois au démarrage de l'application.
    Elles ne sont pas rechargées à chaque interaction utilisateur.
    """

    # Chargement du réseau de neurones entraîné
    keras_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "modele_taekwondo.keras")
    )

    # Chargement du scaler utilisé lors de l'entraînement
    scaler = joblib.load(
        os.path.join(MODEL_DIR, "scaler_taekwondo.pkl")
    )

    # Chargement des noms de classes
    classes = joblib.load(
        os.path.join(MODEL_DIR, "classes.pkl")
    )

    # Chargement de la liste exacte des colonnes utilisées pendant l'entraînement du modèle
    features = joblib.load(
        os.path.join(MODEL_DIR, "feature_columns.pkl")
    )

    return keras_model, scaler, classes, features


# Initialise MediaPipe Pose
@st.cache_resource
def load_pose_landmarker():

    # Chemin du modèle MediaPipe 
    task_path = os.path.join(MODEL_DIR, "pose_landmarker.task")

    # Configuration de base du modèle
    base_options = python.BaseOptions(model_asset_path=task_path)

    # Paramètres de détection de pose
    # On utilise les même paramètres que pour nos données d'entrainements
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, # Traitement image par image
        min_pose_detection_confidence=0.3, # Seuil minimal pour accepter une détection
        min_pose_presence_confidence=0.3, # Seuil minimal de présence du corps détecté       
        min_tracking_confidence=0.3, # Seuil minimal de suivi des landmarks
    )

    # Création de l'objet MediaPipe
    return vision.PoseLandmarker.create_from_options(options)


# Affichage d'un message pendant le chargement initial
with st.spinner("Chargement des modèles…"):
   
    # Chargement du réseau de neurones du scaler, des classes et des colonnes
    keras_model, scaler, classes, feature_columns = load_models()

    # Chargement du détecteur de pose MediaPipe
    landmarker = load_pose_landmarker()


#---------------------------------------
# Fonctions d'extraction et de prédiction
#---------------------------------------

# Détecte la pose sur une image et extrait les coordonnées des landmarks sélectionnés pour le modèle
def extract_landmarks(pil_image):

    # Conversion de l'image PIL en tableau numpy RGB
    img_rgb = np.array(pil_image.convert("RGB"))

    # Conversion au format attendu par MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Détection de la pose
    result = landmarker.detect(mp_image)

    # Si aucun corps n'est détecté  
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None, img_rgb

    # Récupération de la première pose détectée
    landmarks = result.pose_landmarks[0]
    row = {} # Dictionnaire qui contiendra les features
    # Extraction des coordonnées des landmarks conservés
    for idx, name in LANDMARK_NAMES.items():
        lm = landmarks[idx]
        row[f"{name}_x"]   = lm.x
        row[f"{name}_y"]   = lm.y
        row[f"{name}_z"]   = lm.z
        row[f"{name}_vis"] = lm.visibility

    # Dessin des points clés sur l'image
    h, w, _ = img_rgb.shape
    annotated = img_rgb.copy()
    for idx, name in LANDMARK_NAMES.items():
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 6, (230, 57, 70), -1)
        cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 1)

    # Connexions simplifiées (squelette)
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    # Dessin du squelette
    for a, b in connections:
        if a in LANDMARK_NAMES and b in LANDMARK_NAMES:
            la, lb = landmarks[a], landmarks[b]
            p1 = (int(la.x * w), int(la.y * h))
            p2 = (int(lb.x * w), int(lb.y * h))
            cv2.line(annotated, p1, p2, (255, 255, 255), 2)

    return row, annotated

# Prédiction du modèle
def predict(row):
    # Création d'une ligne de coordonnées (même ordre que pendant l'entraînement)
    X = np.array([[row[f] for f in feature_columns]])

    # Application du scaler entraîné
    X_scaled = scaler.transform(X)

    # Prédiction du réseau de neurones
    prob = float(keras_model.predict(X_scaled, verbose=0)[0][0])

    # classes[0] = ap_tchagui (0), classes[1] = yop_tchagui (1)
    if prob >= 0.5:
        label = classes[1]
        confidence = prob
    else:
        label = classes[0]
        confidence = 1 - prob

    return label, confidence, prob