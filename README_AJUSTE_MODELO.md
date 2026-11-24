# Versión ajustada del predictor de concreto

## Qué cambió
- Se conservan 445 registros individuales originales y 20 dosificaciones experimentales en entrenamiento (465 registros).
- Se mantienen 10 dosificaciones experimentales como validación externa aislada.
- Se compararon modelos y ponderaciones mediante validación cruzada agrupada, puntuando el error únicamente sobre las 20 dosificaciones experimentales del entrenamiento.
- Se seleccionó Ridge con alpha=0.1 y peso 15 para cada fila experimental frente a 1 para cada fila original.
- El modelo final está guardado en `models/modelo_ajustado_experimental.joblib`.

## Resultado en las 10 dosificaciones reservadas
- MAE: 2.91 MPa
- RMSE: 3.22 MPa
- Sesgo predicho menos real: +1.92 MPa
- R²: -6.32

El MAE bajó respecto a la primera versión (8.46 MPa), pero R² sigue siendo negativo. Esto significa que el ajuste mejoró el error absoluto medio, pero no se debe afirmar que el modelo ya tenga capacidad predictiva satisfactoria. La validación contiene solo 10 diseños y las resistencias tienen un rango estrecho; ampliar el conjunto independiente es necesario.

## Ejecución
En la carpeta `concreto_ml_peru`:
```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

## Reproducir el ajuste
```powershell
python 10_ajuste_modelo_experimental.py
```
El script no usa las 10 observaciones reservadas para seleccionar hiperparámetros.

## Advertencia
Herramienta académica de investigación. No sustituye ensayos normalizados, validación profesional ni diseño estructural. Verificar los registros originales de laboratorio y las unidades antes de cualquier uso.
