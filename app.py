from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_ajustado_experimental.joblib"
FEATURES = [
    "cemento_kg_m3", "escoria_kg_m3", "fly_ash_kg_m3", "agua_kg_m3",
    "superplastificante_kg_m3", "agregado_grueso_kg_m3",
    "agregado_fino_kg_m3", "edad_dias"
]

@st.cache_resource
def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def mpa_a_kgcm2(mpa):
    return mpa * 10.19716213

st.set_page_config(page_title="Predictor de resistencia del concreto", page_icon="🏗️", layout="wide")
st.title("Predicción de la resistencia a compresión del concreto")
st.write("Ingrese las características de la dosificación y la edad del concreto para obtener una estimación mediante el modelo Ridge ajustado con ponderación de datos experimentales.")

try:
    modelo = cargar_modelo()
except Exception as exc:
    st.error("No se pudo cargar el modelo entrenado. Verifique la carpeta models.")
    st.exception(exc)
    st.stop()

st.subheader("Datos de entrada")
c1, c2 = st.columns(2)
with c1:
    cemento = st.number_input("Cemento (kg/m³)", min_value=0.0, value=350.0, step=1.0)
    escoria = st.number_input("Escoria de alto horno (kg/m³)", min_value=0.0, value=0.0, step=1.0)
    fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0, step=1.0)
    agua = st.number_input("Agua (kg/m³)", min_value=0.0, value=185.0, step=1.0)
with c2:
    superplastificante = st.number_input("Superplastificante (kg/m³)", min_value=0.0, value=0.0, step=0.1)
    agregado_grueso = st.number_input("Agregado grueso (kg/m³)", min_value=0.0, value=1050.0, step=1.0)
    agregado_fino = st.number_input("Agregado fino (kg/m³)", min_value=0.0, value=780.0, step=1.0)
    edad = st.number_input("Edad del concreto (días)", min_value=1.0, value=28.0, step=1.0)

valores = [cemento, escoria, fly_ash, agua, superplastificante, agregado_grueso, agregado_fino, edad]
if st.button("PREDECIR RESISTENCIA", type="primary", use_container_width=True):
    if not all(np.isfinite(v) for v in valores) or any(v < 0 for v in valores[:7]) or edad <= 0:
        st.warning("Revise los valores ingresados. Las dosificaciones no pueden ser negativas y la edad debe ser mayor que cero.")
    else:
        entrada = pd.DataFrame([valores], columns=FEATURES)
        pred_mpa = float(modelo.predict(entrada)[0])
        r1, r2 = st.columns(2)
        r1.metric("Resistencia estimada", f"{pred_mpa:.2f} MPa")
        r2.metric("Resistencia estimada", f"{mpa_a_kgcm2(pred_mpa):.2f} kgf/cm²")
        with st.expander("Ver datos utilizados"):
            st.dataframe(entrada.T.rename(columns={0: "Valor"}), use_container_width=True)

st.divider()
st.subheader("Información y alcance del modelo")
st.write("- Algoritmo: Ridge con ponderación de observaciones experimentales.")
st.write("- Variables de entrada: 8.")
st.write("- Entrenamiento: 445 registros originales + 20 dosificaciones experimentales = 465 registros.")
st.write("- Validación externa reservada: 10 dosificaciones, no utilizadas para entrenar.")
st.warning("En la validación externa reservada (10 dosificaciones) el modelo ajustado obtuvo MAE = 2.91 MPa, RMSE = 3.22 MPa y R² = -6.32. El R² negativo indica que todavía no supera al predictor basado en la media para este pequeño conjunto. Use el resultado solo con fines de investigación; no sustituye ensayos de laboratorio ni sirve para diseño estructural.")
