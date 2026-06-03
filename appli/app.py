# ============================================================
# Projet      : CoachTaekwondo
# Fichier     : app.py
# Auteur      : Aurélie RIVIERE
# Version     : 1.0.0
# Date        : Juin 2026
#
# Description :
# Interface utilisateur Streamlit permettant :
# - le chargement d'images,
# - la détection de pose avec MediaPipe,
# - la classification des techniques de taekwondo,
# - l'affichage des résultats de prédiction.
# ============================================================

# Imports des bibliothèques
import streamlit as st
from PIL import Image

from model_utils import (
    extract_landmarks,
    predict
)

# Chargement CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


#---------------------------------------
# Configuration 
#---------------------------------------
st.set_page_config(
    page_title="CoachTaekwondo",
    page_icon="🥋",
    layout="centered",
)

#---------------------------------------
# En tête 
#---------------------------------------

st.markdown("""
<div class="hero">
    <h1>🥋 CoachTaekwondo</h1>
    <p>Analyse automatique de technique de coup de pied par vision par ordinateur</p>
</div>
""", unsafe_allow_html=True)

# Présentation
with st.expander("📖 À propos de CoachTaekwondo", expanded=False):
    st.markdown("""
    **CoachTaekwondo** est un système d'analyse de techniques de taekwondo basé sur
    l'intelligence artificielle. Il utilise deux technologies combinées :

    - **MediaPipe Pose** (Google) pour détecter les 21 points clés du corps humain sur chaque image
    - **Réseau de neurones** entraîné pour classifier le type de coup de pied

    Le modèle a été entraîné à distinguer deux techniques fondamentales :
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="kick-card">
            <h3>🦵 Ap Tchagui</h3>
            <p>Coup de pied frontal — frappe vers l'avant avec la plante ou le dessus du pied</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kick-card">
            <h3>🦶 Yop Tchagui</h3>
            <p>Coup de pied latéral — frappe sur le côté avec le tranchant du pied</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    **Pipeline de prédiction :**
    1. L'image est analysée par MediaPipe pour extraire 84 coordonnées de points clés
    2. Les coordonnées sont normalisées avec le scaler entraîné
    3. Le réseau de neurones prédit la classe et le niveau de confiance
    """)

st.markdown("---")



#---------------------------------------
# Zone d'upload et prédiction
#---------------------------------------
st.markdown("### 📸 Analyser une image")

# Zone de dépôt de l'image
uploaded_file = st.file_uploader(
    "Déposez une photo d'un pratiquant en train d'exécuter un coup de pied",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

# Exécution uniquement lorsqu'une image est fournie
if uploaded_file:
    pil_image = Image.open(uploaded_file)

    col_img, col_res = st.columns([1, 1])

    # Affichage de l'image d'origine
    with col_img:
        st.markdown("**Image originale**")
        st.image(pil_image, use_container_width=True)

    # Détection de la pose MediaPipe
    with st.spinner("Détection de la pose en cours…"):
        row, annotated = extract_landmarks(pil_image)

    # Cas où aucune pose n'est détectée
    if row is None:
        with col_res:
            st.markdown("""
            <div class="no-detection">
                <b>⚠️ Pose non détectée</b><br><br>
                MediaPipe n'a pas pu localiser de corps humain sur cette image.<br><br>
                <small>Conseils : assurez-vous que le pratiquant est visible en entier,
                l'image est bien éclairée et la personne est de face ou de profil.</small>
            </div>
            """, unsafe_allow_html=True)

    # Cas où la pose est détectée
    else:
        # Affichage de l'image annotée
        with col_res:
            st.markdown("**Pose détectée**")
            st.image(annotated, use_container_width=True)

        # Prédiction du modèle
        label, confidence, prob_yop = predict(row)

        # Affichage du résultat
        is_ap = label == "ap_tchagui"
        css_class = "result-ap" if is_ap else "result-yop"
        display_name = "Ap Tchagui 🦵" if is_ap else "Yop Tchagui 🦶"
        desc = "Coup de pied frontal" if is_ap else "Coup de pied latéral"

        # Conversion en pourcentage
        conf_pct = confidence * 100

        # Affichage du résultat principal
        st.markdown(f"""
        <div class="result-box {css_class}">
            <div class="label">{display_name}</div>
            <div class="conf">{desc}</div>
            <br>
            <div class="conf">Confiance : <b>{conf_pct:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Affichage des probabilités
        st.markdown("**Distribution des probabilités**")
        ap_prob = (1 - prob_yop) * 100
        yop_prob = prob_yop * 100

        # Affichage côte à côte
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Ap Tchagui", f"{ap_prob:.1f}%")
            st.progress(ap_prob / 100)
        with col_b:
            st.metric("Yop Tchagui", f"{yop_prob:.1f}%")
            st.progress(yop_prob / 100)

        # Informations techniques
        st.markdown("#### 🔬 Détails techniques")

        st.markdown(f"""
        <span class="info-chip">84 features extraites</span>
        <span class="info-chip">21 landmarks MediaPipe</span>
        <span class="info-chip">Sortie sigmoid : {prob_yop:.4f}</span>
        <span class="info-chip">Seuil : 0.5</span>
        """, unsafe_allow_html=True)

#---------------------------------------
# Pied de page 
#---------------------------------------

st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:#555;font-size:0.8rem;'>"
    "CoachTaekwondo · MediaPipe + Keras · "
    "Entraîné sur dataset ap_tchagui / yop_tchagui"
    "</p>",
    unsafe_allow_html=True
)