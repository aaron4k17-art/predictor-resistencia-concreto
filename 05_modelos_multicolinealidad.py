from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold,GroupShuffleSplit,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path('/mnt/data/concreto_ml_peru');DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx';RES=BASE/'resultados'
F=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias'];T='resistencia_mpa';G='mezcla_id';N='naturaleza_observacion'
d=pd.read_excel(DATA);d=d[d[N].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in F+[T]:d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.dropna(subset=F+[T,G]).drop_duplicates(subset=F+[T,G]).copy()
sp=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);tr,te=next(sp.split(d[F],d[T],groups=d[G]));train=d.iloc[tr];test=d.iloc[te];cv=GroupKFold(5)
models={
 'PLS optimizado':(Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',PLSRegression(scale=False))]),{'model__n_components':[2,3,4,5,6,7]}),
 'ElasticNet optimizado':(Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',ElasticNet(max_iter=50000,random_state=42))]),{'model__alpha':np.logspace(-3,1,10),'model__l1_ratio':[0.05,0.2,0.5,0.8,0.95]})}
rows=[]
for name,(est,grid) in models.items():
 gs=GridSearchCV(est,grid,cv=cv,scoring='r2',n_jobs=1,refit=True)
 gs.fit(train[F],train[T],groups=train[G]);p=gs.predict(test[F]).ravel()
 rows.append({'modelo':name,'cv_R2':gs.best_score_,'R2_test':r2_score(test[T],p),'MAE_test_MPa':mean_absolute_error(test[T],p),'RMSE_test_MPa':np.sqrt(mean_squared_error(test[T],p)),'mejores_parametros':gs.best_params_})
 print(name,rows[-1])
res=pd.DataFrame(rows);res.to_csv(RES/'modelos_multicolinealidad.csv',index=False,encoding='utf-8-sig')
