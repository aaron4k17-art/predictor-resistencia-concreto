from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE=Path('/mnt/data/concreto_ml_peru'); DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx'; RES=BASE/'resultados'
F=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias']; T='resistencia_mpa'; G='mezcla_id'; N='naturaleza_observacion'

df=pd.read_excel(DATA); df=df[df[N].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in F+[T]: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.dropna(subset=F+[T,G]).drop_duplicates(subset=F+[T,G]).copy()

def p(model,scale=False):
 s=[('imp',SimpleImputer(strategy='median'))]
 if scale:s.append(('sc',StandardScaler()))
 s.append(('model',model));return Pipeline(s)
models={
 'Regresion Lineal':p(LinearRegression(),True),
 'Ridge (alpha=100)':p(Ridge(alpha=100),True),
 'Random Forest (ajustado)':p(RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features='sqrt',max_depth=5,random_state=42,n_jobs=2)),
 'Gradient Boosting (ajustado)':p(GradientBoostingRegressor(n_estimators=300,learning_rate=.02,max_depth=2,min_samples_leaf=1,subsample=.8,random_state=42,loss='squared_error'))
}
rows=[]
seeds=list(range(10))
for seed in seeds:
 sp=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed)
 tr,te=next(sp.split(df[F],df[T],groups=df[G])); train=df.iloc[tr]; test=df.iloc[te]
 for name,m in models.items():
  m.fit(train[F],train[T]); pred=m.predict(test[F])
  rows.append({'seed':seed,'modelo':name,'R2':r2_score(test[T],pred),'MAE_MPa':mean_absolute_error(test[T],pred),'RMSE_MPa':np.sqrt(mean_squared_error(test[T],pred)),'n_test':len(test),'n_grupos_test':test[G].nunique()})
r=pd.DataFrame(rows)
summary=r.groupby('modelo').agg(R2_media=('R2','mean'),R2_std=('R2','std'),R2_min=('R2','min'),R2_max=('R2','max'),MAE_media_MPa=('MAE_MPa','mean'),RMSE_media_MPa=('RMSE_MPa','mean')).reset_index().sort_values('R2_media',ascending=False)
r.to_csv(RES/'estabilidad_10_repeticiones.csv',index=False,encoding='utf-8-sig'); summary.to_csv(RES/'resumen_estabilidad_10_repeticiones.csv',index=False,encoding='utf-8-sig')
print('=== RESUMEN ESTABILIDAD (10 split group-aware) ===');print(summary.to_string(index=False))
print('\n=== R2 POR SEMILLA ===');print(r.pivot(index='seed',columns='modelo',values='R2').round(3).to_string())
