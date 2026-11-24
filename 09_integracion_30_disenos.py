from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / 'datos' / 'DATUM_PARA_ENTRENO.xlsx'
EXPERIMENTAL = BASE / 'datos' / '30_disenos_mezcla_variables_.xlsx'
OUT = BASE / 'resultados'
MODELS = BASE / 'models'
OUT.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
SEED = 42
KGFCM2_PER_MPA = 10.197162129779
FEATURES = ['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3',
            'superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias']
TARGET = 'resistencia_mpa'
GROUP = 'mezcla_id'

# 1) Base documental original: solo observaciones individuales.
df = pd.read_excel(ORIGINAL)
required = FEATURES + [TARGET, GROUP, 'naturaleza_observacion']
missing = [c for c in required if c not in df.columns]
if missing: raise ValueError(f'Faltan columnas en base original: {missing}')
df = df[df['naturaleza_observacion'].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in FEATURES + [TARGET]: df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=FEATURES + [TARGET, GROUP]).drop_duplicates(subset=FEATURES + [TARGET, GROUP]).copy()
df[GROUP] = df[GROUP].astype(str)

# 2) Matriz experimental: el encabezado está en la fila 4 del Excel.
exp = pd.read_excel(EXPERIMENTAL, sheet_name='Matriz_30_mezclas', header=3)
exp = exp.dropna(subset=FEATURES + ['resistencia_kg_cm2']).copy()
for c in FEATURES + ['resistencia_kg_cm2']: exp[c] = pd.to_numeric(exp[c], errors='coerce')
exp = exp.dropna(subset=FEATURES + ['resistencia_kg_cm2']).reset_index(drop=True)
if len(exp) != 30: raise ValueError(f'Se esperaban 30 diseños completos; se encontraron {len(exp)}.')
exp[TARGET] = exp['resistencia_kg_cm2'] / KGFCM2_PER_MPA
exp[GROUP] = [f'EXPERIMENTAL_{i+1:02d}' for i in range(len(exp))]
exp['origen_dato'] = 'Experimental_usuario'
df['origen_dato'] = 'Base_original_individual'

# Comprobar que ningún diseño experimental repita exactamente una dosificación original.
key_original = set(map(tuple, df[FEATURES].round(8).to_numpy()))
key_exp = list(map(tuple, exp[FEATURES].round(8).to_numpy()))
overlap = [i+1 for i, key in enumerate(key_exp) if key in key_original]
if overlap:
    raise ValueError(f'Hay dosificaciones experimentales idénticas a registros originales (filas Excel: {overlap}). Revisar antes de validar.')

# 3) División fija y reproducible: 20 entrenamiento, 10 validación externa.
idx_train, idx_val = train_test_split(np.arange(len(exp)), test_size=10, random_state=SEED, shuffle=True)
exp_train = exp.iloc[idx_train].copy().reset_index(drop=True)
exp_val = exp.iloc[idx_val].copy().reset_index(drop=True)

# Guardar trazabilidad de selección y datos divididos.
exp_train.to_excel(OUT / '20_disenos_agregados_entrenamiento.xlsx', index=False)
exp_val.to_excel(OUT / '10_disenos_validacion_reservada.xlsx', index=False)

train = pd.concat([df, exp_train], ignore_index=True)
X, y, groups = train[FEATURES], train[TARGET], train[GROUP]

# 4) Optimización únicamente con los 465 datos de entrenamiento.
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', ElasticNet(max_iter=100000, random_state=SEED))
])
param_grid = {
    'model__alpha': np.logspace(-3, 0, 15),
    'model__l1_ratio': [0.10,0.20,0.30,0.50,0.70,0.80,0.90,0.95,1.0]
}
cv = GroupKFold(n_splits=5)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring='r2', n_jobs=-1, refit=True)
search.fit(X, y, groups=groups)
model = search.best_estimator_
joblib.dump(model, MODELS / 'elasticnet_entrenado_465_experimental.joblib')

# 5) Evaluación única en las 10 dosificaciones nunca usadas en ajuste/entrenamiento.
y_true = exp_val[TARGET].to_numpy()
y_pred = model.predict(exp_val[FEATURES])
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
metrics = {
    'modelo': 'ElasticNet', 'n_entrenamiento_original': int(len(df)),
    'n_disenos_experimentales_agregados': int(len(exp_train)),
    'n_total_entrenamiento': int(len(train)), 'n_validacion_externa': int(len(exp_val)),
    'n_mezclas_grupos_entrenamiento': int(train[GROUP].nunique()),
    'semilla_seleccion': SEED, 'r2_validacion_10': float(r2_score(y_true, y_pred)),
    'mae_mpa': float(mean_absolute_error(y_true, y_pred)), 'rmse_mpa': rmse,
    'sesgo_predicho_menos_real_mpa': float(np.mean(y_pred-y_true)),
    'hiperparametros': {k: (float(v) if isinstance(v, (np.floating,)) else v) for k,v in search.best_params_.items()},
    'r2_cv_groupkfold_5_entrenamiento': float(search.best_score_),
    'nota': 'Las 10 filas de validación no se usaron para entrenar ni seleccionar hiperparámetros.'
}
with open(OUT / 'metricas_validacion_experimental.json','w',encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

pred = exp_val[FEATURES + ['resistencia_kg_cm2', TARGET, GROUP]].copy()
pred['resistencia_real_kg_cm2'] = pred['resistencia_kg_cm2']
pred['resistencia_predicha_mpa'] = y_pred
pred['resistencia_predicha_kg_cm2'] = y_pred * KGFCM2_PER_MPA
pred['error_mpa_predicho_menos_real'] = y_pred-y_true
pred['error_absoluto_mpa'] = np.abs(y_pred-y_true)
pred.to_excel(OUT / 'predicciones_validacion_10_disenos.xlsx', index=False)

# Resumen reproducible de los datos de entrenamiento y validación.
train.to_excel(OUT / 'base_entrenamiento_465_registros.xlsx', index=False)
config = {
    'modelo_final':'ElasticNet', 'variables_entrada':FEATURES, 'variable_objetivo':TARGET,
    'grupo':GROUP, 'entrenamiento':{'base_original_individual':len(df),'nuevos_disenos':len(exp_train),'total':len(train)},
    'validacion_externa':len(exp_val), 'semilla':SEED,
    'metodo_cv':'GroupKFold de 5 particiones, agrupado por mezcla_id',
    'preprocesamiento':'SimpleImputer(mediana) + StandardScaler',
    'mejores_hiperparametros':metrics['hiperparametros'],
    'r2_cv':float(search.best_score_),
    'unidad_origen_resistencia_experimental':'kgf/cm²',
    'conversion':'MPa = kgf/cm² / 10.197162129779',
    'validacion_aislada':True
}
with open(OUT / 'configuracion_integracion_30_disenos.json','w',encoding='utf-8') as f:
    json.dump(config,f,ensure_ascii=False,indent=2)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print('\nValidación (10 diseños):')
print(pred[['resistencia_real_kg_cm2','resistencia_predicha_kg_cm2','error_absoluto_mpa']].to_string(index=False))
