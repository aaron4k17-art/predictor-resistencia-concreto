from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')
BASE=Path(__file__).resolve().parent; OUT=BASE/'resultados'; MODELS=BASE/'models'
FEATURES=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias']
TARGET='resistencia_mpa'; GROUP='mezcla_id'; SEED=42
train=pd.read_excel(OUT/'base_entrenamiento_465_registros.xlsx')
val=pd.read_excel(OUT/'10_disenos_validacion_reservada.xlsx')
# Comparación interna: predicciones fuera de pliegue solo en los 20 experimentos de entrenamiento.
gkf=GroupKFold(n_splits=3)
exper_idx=np.where(train['origen_dato'].eq('Experimental_usuario').to_numpy())[0]
configs=[]
for w in [1,5,15]:
 for alpha in [0.01,0.1,1,10]:
  for l1 in [0.2,0.8,1.0]:
   configs.append((f'ElasticNet_w{w}_a{alpha}_l{l1}', 'elastic', w, alpha, l1))
 for alpha in [0.1,1,10,100]: configs.append((f'Ridge_w{w}_a{alpha}','ridge',w,alpha,None))
# Árboles también reciben peso experimental; limitar complejidad por muestra pequeña.
for w in [1,5,15]:
 for depth in [2,3]:
  configs.append((f'RandomForest_w{w}_d{depth}','rf',w,depth,None))
  configs.append((f'ExtraTrees_w{w}_d{depth}','et',w,depth,None))

rows=[]
for name,kind,w,a,l1 in configs:
 oof=np.full(len(train),np.nan)
 for tr,te in gkf.split(train[FEATURES],train[TARGET],groups=train[GROUP]):
  Xtr=train.iloc[tr][FEATURES]; ytr=train.iloc[tr][TARGET]
  expmask=train.iloc[tr]['origen_dato'].eq('Experimental_usuario').to_numpy()
  sw=np.where(expmask,w,1.0)
  if kind=='elastic': est=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',ElasticNet(alpha=a,l1_ratio=l1,max_iter=200000,random_state=SEED))]); est.fit(Xtr,ytr,m__sample_weight=sw)
  elif kind=='ridge': est=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',Ridge(alpha=a))]); est.fit(Xtr,ytr,m__sample_weight=sw)
  elif kind=='rf': est=Pipeline([('imp',SimpleImputer(strategy='median')),('m',RandomForestRegressor(n_estimators=100,max_depth=a,min_samples_leaf=2,max_features=0.9,random_state=SEED,n_jobs=-1))]); est.fit(Xtr,ytr,m__sample_weight=sw)
  else: est=Pipeline([('imp',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=100,max_depth=a,min_samples_leaf=2,max_features=0.9,random_state=SEED,n_jobs=-1))]); est.fit(Xtr,ytr,m__sample_weight=sw)
  oof[te]=est.predict(train.iloc[te][FEATURES])
 # score only OOF experimental observations; original dataset metrics separately
 yt=train.iloc[exper_idx][TARGET].to_numpy(); yp=oof[exper_idx]
 mae=mean_absolute_error(yt,yp); rmse=np.sqrt(mean_squared_error(yt,yp)); r2=r2_score(yt,yp)
 rows.append({'config':name,'tipo':kind,'peso_experimental':w,'alpha_o_profundidad':a,'l1_ratio':l1,'mae_cv_experimental_mpa':mae,'rmse_cv_experimental_mpa':rmse,'r2_cv_experimental':r2})
res=pd.DataFrame(rows).sort_values(['mae_cv_experimental_mpa','rmse_cv_experimental_mpa'])
res.to_csv(OUT/'comparacion_ajuste_cv_experimental.csv',index=False)
best=res.iloc[0].to_dict(); name=best['config']; parts=name.split('_')
w=float(best['peso_experimental']); kind=best['tipo']; a=float(best['alpha_o_profundidad']); l1=best['l1_ratio']
if kind=='elastic': model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',ElasticNet(alpha=a,l1_ratio=float(l1),max_iter=200000,random_state=SEED))])
elif kind=='ridge': model=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',Ridge(alpha=a))])
elif kind=='rf': model=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=250,max_depth=int(a),min_samples_leaf=2,max_features=0.9,random_state=SEED,n_jobs=-1))])
else: model=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',ExtraTreesRegressor(n_estimators=250,max_depth=int(a),min_samples_leaf=2,max_features=0.9,random_state=SEED,n_jobs=-1))])
sw=np.where(train['origen_dato'].eq('Experimental_usuario').to_numpy(),w,1.0)
if kind in ('elastic','ridge'): model.fit(train[FEATURES],train[TARGET],model__sample_weight=sw)
else: model.fit(train[FEATURES],train[TARGET],model__sample_weight=sw)
# Única evaluación externa final, sin selección usando los 10 resultados.
ytrue=val[TARGET].to_numpy(); ypred=model.predict(val[FEATURES]); rmse=float(np.sqrt(mean_squared_error(ytrue,ypred)))
metrics={'modelo_seleccionado':name,'criterio_seleccion':'menor MAE de validación cruzada agrupada calculado exclusivamente sobre los 20 diseños experimentales del entrenamiento','peso_experimental_vs_original':w,'cv_experimental_mae_mpa':float(best['mae_cv_experimental_mpa']),'cv_experimental_rmse_mpa':float(best['rmse_cv_experimental_mpa']),'cv_experimental_r2':float(best['r2_cv_experimental']),'validacion_externa_n':len(val),'validacion_externa_r2':float(r2_score(ytrue,ypred)),'validacion_externa_mae_mpa':float(mean_absolute_error(ytrue,ypred)),'validacion_externa_rmse_mpa':rmse,'validacion_externa_sesgo_mpa':float(np.mean(ypred-ytrue)),'nota':'Los 10 diseños externos no se utilizaron en la selección ni el ajuste. La validación es pequeña (n=10), por lo que las métricas son preliminares.'}
joblib.dump(model,MODELS/'modelo_ajustado_experimental.joblib')
with open(OUT/'metricas_modelo_ajustado.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
pred=val.copy(); pred['prediccion_mpa']=ypred; pred['prediccion_kgf_cm2']=ypred*10.197162129779; pred['error_mpa']=ypred-ytrue; pred['error_absoluto_mpa']=np.abs(ypred-ytrue)
pred.to_excel(OUT/'validacion_externa_modelo_ajustado.xlsx',index=False)
print(json.dumps(metrics,ensure_ascii=False,indent=2)); print('\nTop 10 CV experimental:'); print(res.head(10).to_string(index=False)); print('\nValidación externa:'); print(pred[['resistencia_kg_cm2','prediccion_kgf_cm2','error_absoluto_mpa']].to_string(index=False))
