# Machine Learning - Resistencia del concreto - Peru

## Paso 1: entrenamiento base

La base `DATUM_PARA_ENTRENO.xlsx` contiene registros documentales de estudios peruanos. Para el modelo principal se utilizan solamente las observaciones marcadas como `INDIVIDUAL`.

El script:

1. Carga y valida columnas.
2. Filtra observaciones individuales.
3. Elimina filas incompletas en las variables del modelo.
4. Elimina duplicados exactos.
5. Divide train/test por `mezcla_id` con `GroupShuffleSplit` para evitar fuga entre probetas de una misma mezcla.
6. Entrena Regresion Lineal, Random Forest y Gradient Boosting.
7. Calcula R2, MAE y RMSE en test.
8. Ejecuta 5-Fold `GroupKFold`.
9. Selecciona el mejor modelo por R2 medio de CV y usa RMSE CV como desempate.
10. Guarda modelos y resultados para utilizarlos despues en el software.

## Ejecucion en VS Code

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Luego:

```bash
pip install -r requirements.txt
python 01_entrenamiento.py
```

Los resultados aparecen en `resultados/` y los modelos en `models/`.


## Ejecutar el software de predicción

En Windows, abra esta carpeta en VS Code o PowerShell y ejecute:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación carga `models/elasticnet_entrenado_465_experimental.joblib`. Este modelo incorpora los 20 diseños seleccionados al entrenamiento (465 registros en total); los otros 10 diseños se reservaron para validación externa. La validación inicial no fue satisfactoria (R² = -63.883; MAE = 8.46 MPa), por lo que la aplicación es únicamente de investigación y no debe emplearse para decisiones de diseño estructural.
