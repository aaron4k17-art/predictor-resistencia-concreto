from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path('/mnt/data/concreto_ml_peru'); DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx'; RES=BASE/'resultados'; MOD=BASE/'models'
FEATURES=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias']
TARGET='resistencia_mpa'; GROUP='mezcla_id'; NATURE='naturaleza_observacion'

df=pd.read_excel(DATA)
df=df[df[NATURE].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in FEATURES+[TARGET]: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=FEATURES+[TARGET,GROUP]).drop_duplicates(subset=FEATURES+[TARGET,GROUP]).copy()

pipe=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler()),
    ('model',ElasticNet(max_iter=50000,random_state=42))
])
param_grid={
    'model__alpha':np.logspace(-3,0,12),
    'model__l1_ratio':[0.10,0.20,0.30,0.50,0.70,0.80,0.90,0.95]
}
cv=GroupKFold(n_splits=5)
search=GridSearchCV(pipe,param_grid,cv=cv,scoring='r2',n_jobs=1,refit=True,return_train_score=False)
search.fit(df[FEATURES],df[TARGET],groups=df[GROUP])
final_model=search.best_estimator_
joblib.dump(final_model,MOD/'mejor_modelo_final_elasticnet.joblib')
# Para el software, mantener el nombre genérico apuntando al modelo final.
joblib.dump(final_model,MOD/'mejor_modelo.joblib')

cfg={
 'modelo_final':'ElasticNet',
 'n_registros':int(len(df)),
 'n_mezclas':int(df[GROUP].nunique()),
 'features':FEATURES,
 'target':TARGET,
 'grupo':GROUP,
 'validacion_para_ajuste':{'metodo':'GroupKFold','n_splits':5},
 'seleccion_hiperparametros':{'criterio':'mayor R2 medio en GroupKFold','mejor_R2_CV':float(search.best_score_)},
 'hiperparametros':{k: (float(v) if isinstance(v,(np.floating,)) else v) for k,v in search.best_params_.items()},
 'preprocesamiento':'SimpleImputer(mediana) + StandardScaler'
}
with open(RES/'configuracion_modelo_final.json','w',encoding='utf-8') as f: json.dump(cfg,f,ensure_ascii=False,indent=2)
print('MODELO FINAL')
print('ElasticNet')
print('R2 CV:',round(search.best_score_,6))
print('Parametros:',search.best_params_)
print('Registros:',len(df),'Mezclas:',df[GROUP].nunique())
