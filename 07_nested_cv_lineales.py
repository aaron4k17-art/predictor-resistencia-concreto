from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
BASE=Path('/mnt/data/concreto_ml_peru');DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx';RES=BASE/'resultados'
F=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias'];T='resistencia_mpa';G='mezcla_id';N='naturaleza_observacion'
d=pd.read_excel(DATA);d=d[d[N].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in F+[T]:d[c]=pd.to_numeric(d[c],errors='coerce')
d=d.dropna(subset=F+[T,G]).drop_duplicates(subset=F+[T,G]).copy()
X=d[F];y=d[T];groups=d[G]
outer=GroupKFold(5);rows=[]
for fold,(tr,te) in enumerate(outer.split(X,y,groups),1):
 Xtr,Xte=X.iloc[tr],X.iloc[te];ytr,yte=y.iloc[tr],y.iloc[te];gtr=groups.iloc[tr]
 models={}
 linear=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',LinearRegression())]);models['Regresion Lineal']=linear
 pls=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',PLSRegression(scale=False))])
 pgrid=GridSearchCV(pls,{'model__n_components':[2,3,4,5,6,7]},cv=GroupKFold(4),scoring='r2',n_jobs=1,refit=True);pgrid.fit(Xtr,ytr,groups=gtr);models['PLS']=pgrid.best_estimator_
 en=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',ElasticNet(max_iter=50000,random_state=42))])
 egrid=GridSearchCV(en,{'model__alpha':np.logspace(-3,0,8),'model__l1_ratio':[0.1,0.3,0.5,0.7,0.9,0.95]},cv=GroupKFold(4),scoring='r2',n_jobs=1,refit=True);egrid.fit(Xtr,ytr,groups=gtr);models['ElasticNet']=egrid.best_estimator_
 for name,m in models.items():
  m.fit(Xtr,ytr);pred=np.asarray(m.predict(Xte)).ravel();rows.append({'outer_fold':fold,'modelo':name,'R2':r2_score(yte,pred),'MAE_MPa':mean_absolute_error(yte,pred),'RMSE_MPa':np.sqrt(mean_squared_error(yte,pred)),'n_test':len(te),'grupos_test':groups.iloc[te].nunique(),'best_params':str((pgrid.best_params_ if name=='PLS' else egrid.best_params_ if name=='ElasticNet' else {}))})
r=pd.DataFrame(rows);s=r.groupby('modelo').agg(R2_media=('R2','mean'),R2_std=('R2','std'),R2_min=('R2','min'),R2_max=('R2','max'),MAE_media_MPa=('MAE_MPa','mean'),RMSE_media_MPa=('RMSE_MPa','mean')).reset_index().sort_values('R2_media',ascending=False)
r.to_csv(RES/'nested_cv_detalle.csv',index=False,encoding='utf-8-sig');s.to_csv(RES/'nested_cv_resumen.csv',index=False,encoding='utf-8-sig');print(s.to_string(index=False));print('\nDetalle:');print(r.to_string(index=False))
