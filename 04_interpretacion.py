from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM=True
except Exception:
    HAS_SM=False

BASE=Path('/mnt/data/concreto_ml_peru'); DATA=BASE/'datos'/'DATUM_PARA_ENTRENO.xlsx'; RES=BASE/'resultados'
F=['cemento_kg_m3','escoria_kg_m3','fly_ash_kg_m3','agua_kg_m3','superplastificante_kg_m3','agregado_grueso_kg_m3','agregado_fino_kg_m3','edad_dias'];T='resistencia_mpa';G='mezcla_id';N='naturaleza_observacion'
D=pd.read_excel(DATA);D=D[D[N].astype(str).str.upper().eq('INDIVIDUAL')].copy()
for c in F+[T]:D[c]=pd.to_numeric(D[c],errors='coerce')
D=D.dropna(subset=F+[T,G]).drop_duplicates(subset=F+[T,G]).copy()
sp=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42);tr,te=next(sp.split(D[F],D[T],groups=D[G]));train=D.iloc[tr];test=D.iloc[te]
model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('model',LinearRegression())]);model.fit(train[F],train[T]);pred=model.predict(test[F])
coef=model.named_steps['model'].coef_; imp=pd.DataFrame({'variable':F,'coef_estandarizado':coef,'abs_coef':np.abs(coef)}).sort_values('abs_coef',ascending=False);imp.to_csv(RES/'coeficientes_lineal_estandarizados.csv',index=False,encoding='utf-8-sig')
perm=permutation_importance(model,test[F],test[T],n_repeats=30,random_state=42,scoring='r2')
pd.DataFrame({'variable':F,'perm_importancia_R2_media':perm.importances_mean,'perm_importancia_R2_std':perm.importances_std}).sort_values('perm_importancia_R2_media',ascending=False).to_csv(RES/'importancia_permutacion_lineal_test.csv',index=False,encoding='utf-8-sig')
res=pd.DataFrame({'real_MPa':test[T].values,'predicho_MPa':pred,'residuo_MPa':test[T].values-pred,'mezcla_id':test[G].values});res.to_csv(RES/'residuos_test_lineal.csv',index=False,encoding='utf-8-sig')
print('=== DESEMPEÑO TEST ===');print({'R2':r2_score(test[T],pred),'MAE':mean_absolute_error(test[T],pred),'RMSE':np.sqrt(mean_squared_error(test[T],pred))})
print('\n=== COEFICIENTES ESTANDARIZADOS ===');print(imp.to_string(index=False))
print('\n=== IMPORTANCIA POR PERMUTACION (TEST) ===');print(pd.read_csv(RES/'importancia_permutacion_lineal_test.csv').to_string(index=False))
if HAS_SM:
    Xi=SimpleImputer(strategy='median').fit_transform(D[F]); V=pd.DataFrame({'variable':F,'VIF':[variance_inflation_factor(Xi,i) for i in range(Xi.shape[1])]});V.to_csv(RES/'vif_variables.csv',index=False,encoding='utf-8-sig');print('\n=== VIF ===');print(V.sort_values('VIF',ascending=False).to_string(index=False))
