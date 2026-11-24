# Integración de 30 diseños experimentales

## División aplicada
- 445 observaciones individuales de la base original.
- 20 diseños seleccionados con `random_state=42` agregados al entrenamiento.
- Total de entrenamiento: 465 registros.
- 10 diseños reservados para validación externa.
- Los 10 registros de validación no intervienen en entrenamiento ni en selección de hiperparámetros.

## Ejecución
Desde la carpeta del proyecto:

```bash
pip install -r requirements.txt
python 09_integracion_30_disenos.py
```

El script entrena ElasticNet con imputación por mediana, estandarización y búsqueda de hiperparámetros mediante GroupKFold (5 particiones), agrupando por mezcla. Las resistencias experimentales del Excel se convierten de kgf/cm² a MPa.

## Resultado inicial
- R² en las 10 dosificaciones reservadas: -63.8830
- MAE: 8.4620 MPa
- RMSE: 9.5891 MPa
- Sesgo medio (predicho - real): +7.9685 MPa

El resultado indica un desempeño deficiente en esta validación; no se debe afirmar que el modelo quedó validado satisfactoriamente ni usar este modelo como predictor confiable sin investigar la causa y mejorar la representatividad de los datos.

## Advertencia de trazabilidad
La hoja `Criterios` del archivo `30_disenos_mezcla_variables_.xlsx` describe la columna `resistencia_kg_cm2` como “Valores simulados para ilustrar la matriz” y dice que no son resultados medidos de prensa. El usuario ha indicado expresamente que los valores sí corresponden a ensayos reales. Esta contradicción debe resolverse con registros de laboratorio (identificación de probetas, fecha, carga máxima, dimensiones, norma/procedimiento y reporte de prensa) antes de presentar los resultados como evidencia experimental.

Además, los nuevos diseños tienen rangos de agregados y dosificaciones distintos a los de la base documental original. La distribución de datos cambia; el rendimiento bajo en las 10 filas de validación debe analizarse y no ocultarse.

El modelo generado se guarda como `models/elasticnet_entrenado_465_experimental.joblib` y no reemplaza el modelo anterior de la aplicación, precisamente porque su validación fue deficiente.
