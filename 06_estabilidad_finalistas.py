from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
BASE=Path('/mnt/data/concreto_ml_peru');DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx';RES=BASE/'resultados'
F=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias'];T='resistencia_mpa';G='mezcla_id';N='naturaleza_observacion'
d=pd.read_excel(DATA);d=d[d[N].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in F+[T]:d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.dropna(subset=F+[T,G]).drop_duplicates(subset=F+[T,G]).copy()
def p(m): return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',m)])
models={
 'Regresion Lineal':p(LinearRegression()),
 'PLS (6 componentes)':p(PLSRegression(n_components=6,scale=False)),
 'ElasticNet':p(ElasticNet(alpha=0.1668100537200059,l1_ratio=.95,max_iter=50000,random_state=42)),
 'Gradient Boosting':p(GradientBoostingRegressor(n_estimators=300,learning_rate=.02,max_depth=2,min_samples_leaf=1,subsample=.8,random_state=42,loss='squared_error')),
 'Random Forest':Pipeline([('imp',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features='sqrt',max_depth=5,random_state=42,n_jobs=2))])}
rows=[]
for seed in range(10):
 sp=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed);tr,te=next(sp.split(d[F],d[T],groups=d[G]));train=d.iloc[tr];test=d.iloc[te]
 for name,m in models.items():
  m.fit(train[F],train[T]);pred=np.asarray(m.predict(test[F])).ravel()
  rows.append({'seed':seed,'modelo':name,'R2':r2_score(test[T],pred),'MAE_MPa':mean_absolute_error(test[T],pred),'RMSE_MPa':np.sqrt(mean_squared_error(test[T],pred))})
r=pd.DataFrame(rows);s=r.groupby('modelo').agg(R2_media=('R2','mean'),R2_std=('R2','std'),R2_min=('R2','min'),R2_max=('R2','max'),MAE_media_MPa=('MAE_MPa','mean'),RMSE_media_MPa=('RMSE_MPa','mean')).sort_values('R2_media',ascending=False).reset_index();r.to_csv(RES/'estabilidad_finalistas_10_repeticiones.csv',index=False,encoding='utf-8-sig');s.to_csv(RES/'resumen_finalistas_10_repeticiones.csv',index=False,encoding='utf-8-sig');print(s.to_string(index=False))
