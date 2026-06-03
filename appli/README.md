# 🥋 CoachTaekwondo

Application Streamlit de classification de coups de pied en taekwondo par vision par ordinateur.

## Technologies

- **MediaPipe Pose** — détection des points clés du corps
- **TensorFlow / Keras** — réseau de neurones de classification
- **Streamlit** — interface web interactive

## Classes détectées

| Classe | Description |
|--------|-------------|
| `ap_tchagui` | Coup de pied frontal |
| `yop_tchagui` | Coup de pied latéral |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Fichiers nécessaires

app.py
requirements.txt
modele_taekwondo.keras      ← modèle Keras entraîné
pose_landmarker.task        ← modèle MediaPipe
scaler_taekwondo.pkl        ← StandardScaler
classes.pkl                 ← liste des classes
feature_columns.pkl         ← noms des features


## Pipeline de prédiction

1. Upload d'une image JPG/PNG
2. MediaPipe extrait 84 coordonnées (21 points × 4 : x, y, z, visibilité)
3. Le StandardScaler normalise les features
4. Le réseau de neurones prédit la classe avec un score de confiance
5. L'image annotée avec le squelette est affichée