from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

BASE=Path('/mnt/data/concreto_ml_peru'); DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx'; RES=BASE/'resultados'; MOD=BASE/'models'
FEATURES=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias']
TARGET='resistencia_mpa'; GROUP='mezcla_id'; NATURE='naturaleza_observacion'; SEED=42

def load():
    d=pd.read_excel(DATA)
    d=d[d[NATURE].astype(str).str.upper().eq('INDIVIDUAL')].copy()
    for c in FEATURES+[TARGET]: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=FEATURES+[TARGET,GROUP]).drop_duplicates(subset=FEATURES+[TARGET,GROUP]).copy()
    return d

def pipe(model, scale=False):
    steps=[('imp',SimpleImputer(strategy='median'))]
    if scale: steps.append(('sc',StandardScaler()))
    steps.append(('model',model)); return Pipeline(steps)

df=load(); split=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=SEED)
tr,te=next(split.split(df[FEATURES],df[TARGET],groups=df[GROUP])); train=df.iloc[tr].copy(); test=df.iloc[te].copy()
Xtr,ytr=train[FEATURES],train[TARGET]; Xte,yte=test[FEATURES],test[TARGET]; g=train[GROUP]
cv=GroupKFold(n_splits=5)
print(f'Individuales={len(df)}, grupos={df[GROUP].nunique()}, train={len(train)}, test={len(test)}')

searches={
'Ridge optimizado': (pipe(Ridge()), {'model__alpha':[0.1,1,10,100]}),
'Random Forest optimizado': (pipe(RandomForestRegressor(random_state=SEED,n_jobs=2)), {
 'model__n_estimators':[100,200,300], 'model__max_depth':[None,5,10], 'model__min_samples_leaf':[1,2,4], 'model__max_features':['sqrt',0.8,1.0]}),
'Gradient Boosting optimizado': (pipe(GradientBoostingRegressor(random_state=SEED,loss='squared_error')), {
 'model__n_estimators':[100,200,300], 'model__learning_rate':[0.02,0.03,0.05], 'model__max_depth':[2,3], 'model__min_samples_leaf':[1,2,4], 'model__subsample':[0.8,1.0]})}

models={'Regresion Lineal':pipe(LinearRegression(),scale=True)}; meta=[]
for name,(est,dist) in searches.items():
    print('ajustando',name,flush=True)
    n_iter=4 if name=='Ridge optimizado' else 8
    rs=RandomizedSearchCV(est,dist,n_iter=n_iter,scoring='r2',cv=cv,random_state=SEED,n_jobs=2,refit=True,return_train_score=False)
    rs.fit(Xtr,ytr,groups=g); models[name]=rs.best_estimator_; meta.append((name,rs.best_score_,rs.best_params_)); print('  cv_r2=',round(rs.best_score_,4),rs.best_params_,flush=True)

models_all={**models}
rows=[]
for name,m in models_all.items():
    m.fit(Xtr,ytr); p=m.predict(Xte)
    rows.append({'modelo':name,'R2_test':r2_score(yte,p),'MAE_test_MPa':mean_absolute_error(yte,p),'RMSE_test_MPa':np.sqrt(mean_squared_error(yte,p))})
res=pd.DataFrame(rows).sort_values('R2_test',ascending=False)
print('\nRESULTADOS TEST'); print(res.to_string(index=False))
# Save selected by TEST only as exploratory ranking; final selection should be justified with CV + test.
res.to_csv(RES/'metricas_ajuste_rapido_test.csv',index=False,encoding='utf-8-sig')
with open(RES/'hiperparametros_ajuste_rapido.json','w') as f: json.dump({'tuning_cv':meta},f,ensure_ascii=False,indent=2,default=str)
# Choose by highest tuning CV R2, then evaluate already fitted model on test.
best_name=max([(n,s) for n,s,p in meta],key=lambda x:x[1])[0] if meta else 'Regresion Lineal'
if best_name in models: best=models[best_name]
else: best=models_all['Regresion Lineal']
best.fit(df[FEATURES],df[TARGET]); joblib.dump(best,MOD/'mejor_modelo_optimizado.joblib')
print('\nBEST_BY_TUNING=',best_name)
