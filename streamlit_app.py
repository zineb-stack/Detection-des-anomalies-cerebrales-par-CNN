"""NeuroDetect, démo web (Streamlit).

Version en ligne de l'application de bureau NeuroDetect :
on importe une image IRM et le modèle MobileNetV2 prédit sa classe.
Lancer en local :  streamlit run streamlit_app.py
"""
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
MODEL_CANDIDATES = [ROOT / "app" / "ui" / "best_model.keras", ROOT / "best_model.keras"]
ENCODER_PATH = ROOT / "label_encoder.pkl"
IMG_SIZE = 224

CLASSES = {
    "glioma":     {"label": "Gliome",              "zone": "Tissu glial cérébral",            "color": "#ff4d6d", "tumor": True},
    "meningioma": {"label": "Méningiome",          "zone": "Méninges (enveloppes du cerveau)", "color": "#fbbf24", "tumor": True},
    "pituitary":  {"label": "Tumeur hypophysaire", "zone": "Glande hypophysaire",             "color": "#a78bfa", "tumor": True},
    "notumor":    {"label": "Aucune tumeur",       "zone": "—",                               "color": "#34d399", "tumor": False},
}

st.set_page_config(page_title="NeuroDetect | Démo", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Chargement du modèle MobileNetV2…")
def load_model():
    import keras  # import ici : TensorFlow est long à charger

    path = next(p for p in MODEL_CANDIDATES if p.exists())
    model = keras.saving.load_model(path, compile=False)
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    try:
        with open(ENCODER_PATH, "rb") as f:
            classes = [str(c) for c in pickle.load(f).classes_]
    except Exception:
        pass  # on garde l'ordre par défaut, identique à l'application
    return model, classes


def preprocess(img: Image.Image) -> np.ndarray:
    # Même préparation que l'application de bureau : RGB, 224x224, valeurs entre 0 et 1
    arr = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32") / 255.0
    return arr[np.newaxis, ...]


# ---------------- page ----------------
st.title("🧠 NeuroDetect")
st.markdown(
    "Détection de tumeurs cérébrales sur IRM avec un réseau de neurones convolutif "
    "(**MobileNetV2**). Importez une image IRM : le modèle la classe parmi "
    "**gliome, méningiome, tumeur hypophysaire** ou **aucune tumeur**."
)
st.info(
    "Projet académique de démonstration. Ce n'est pas un outil médical "
    "et il ne remplace pas l'avis d'un médecin.",
    icon="ℹ️",
)

file = st.file_uploader("Image IRM (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if file is None:
    st.caption("Astuce : utilisez une coupe IRM du cerveau, comme celles du jeu de données d'entraînement.")
else:
    image = Image.open(file)
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.image(image, caption=file.name, use_container_width=True)

    with right:
        try:
            model, classes = load_model()
        except Exception as e:
            st.error(f"Impossible de charger le modèle : {e}")
            st.stop()

        with st.spinner("Analyse en cours…"):
            probs = model.predict(preprocess(image), verbose=0)[0]

        best = int(np.argmax(probs))
        key = classes[best]
        info = CLASSES.get(key, {"label": key, "zone": "—", "color": "#48ceba", "tumor": True})

        st.markdown(
            f"<div style='border-left:6px solid {info['color']};padding:12px 16px;"
            f"border-radius:10px;background:rgba(127,127,127,.08)'>"
            f"<div style='font-size:14px;opacity:.75'>Résultat du modèle</div>"
            f"<div style='font-size:28px;font-weight:700;color:{info['color']}'>{info['label']}</div>"
            f"<div style='font-size:15px'>Confiance : <b>{probs[best] * 100:.1f} %</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if info["tumor"]:
            st.caption(f"Zone concernée : {info['zone']}")

        st.markdown("**Probabilités par classe**")
        for i in np.argsort(probs)[::-1]:
            name = CLASSES.get(classes[i], {"label": classes[i]})["label"]
            st.progress(float(probs[i]), text=f"{name} : {probs[i] * 100:.1f} %")

with st.sidebar:
    st.header("À propos")
    st.markdown(
        "- **Modèle** : MobileNetV2 (transfer learning)\n"
        "- **Entrée** : image 224 × 224, RGB\n"
        "- **Classes** : 4\n"
        "- **Outils** : Python, TensorFlow / Keras\n\n"
        "Projet réalisé en équipe avec [NohaBT](https://github.com/NohaBT).\n\n"
        "Version complète : application de bureau (PySide6) avec espaces médecin, "
        "étudiant et patient.\n\n"
        "[Portfolio de Zineb Ennakhla](https://zineb-portfolio.streamlit.app)"
    )
