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

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datos" / "DATUM_PARA_ENTRENO.xlsx"
RESULTS_DIR = BASE_DIR / "resultados"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
FEATURES = [
    "cemento_kg_m3", "escoria_kg_m3", "fly_ash_kg_m3", "agua_kg_m3",
    "superplastificante_kg_m3", "agregado_grueso_kg_m3", "agregado_fino_kg_m3", "edad_dias"
]
TARGET = "resistencia_mpa"
GROUP_COL = "mezcla_id"
NATURE_COL = "naturaleza_observacion"


def cargar_datos(path):
    df = pd.read_excel(path)
    required = FEATURES + [TARGET, GROUP_COL, NATURE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    df = df[df[NATURE_COL].astype(str).str.upper().eq("INDIVIDUAL")].copy()
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()
    df = df.drop_duplicates(subset=FEATURES + [TARGET, GROUP_COL]).copy()
    return df


def construir_base():
    imputer = SimpleImputer(strategy="median")
    return imputer


def main():
    df = cargar_datos(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP_COL]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    idx_train, idx_test = next(splitter.split(X, y, groups=groups))
    train_df = df.iloc[idx_train].copy()
    test_df = df.iloc[idx_test].copy()

    X_train, y_train, g_train = train_df[FEATURES], train_df[TARGET], train_df[GROUP_COL]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    cv = GroupKFold(n_splits=N_SPLITS)
    scoring = "r2"

    print("=== DATOS ===")
    print(f"Individuales: {len(df)}")
    print(f"Grupos: {df[GROUP_COL].nunique()}")
    print(f"Train: {len(train_df)} | grupos={train_df[GROUP_COL].nunique()}")
    print(f"Test:  {len(test_df)} | grupos={test_df[GROUP_COL].nunique()}")
    print()

    # ========================================================
    # 1. MODELOS BASE DE REFERENCIA
    # ========================================================
    linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    # ========================================================
    # 2. BÚSQUEDA DE HIPERPARÁMETROS SOLO EN TRAIN
    # ========================================================
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    rf_params = {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, 0.8, "sqrt"],
    }

    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error"))
    ])
    gb_params = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.8, 1.0],
    }

    print("=== AJUSTE RANDOM FOREST ===")
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_params,
        n_iter=12,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    rf_search.fit(X_train, y_train, groups=g_train)
    print(f"Mejor R2 CV RF: {rf_search.best_score_:.4f}")
    print(f"Mejores parámetros RF: {rf_search.best_params_}")
    print()

    print("=== AJUSTE GRADIENT BOOSTING ===")
    gb_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=gb_params,
        n_iter=12,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    gb_search.fit(X_train, y_train, groups=g_train)
    print(f"Mejor R2 CV GB: {gb_search.best_score_:.4f}")
    print(f"Mejores parámetros GB: {gb_search.best_params_}")
    print()

    print("=== AJUSTE RIDGE ===")
    ridge_search = RandomizedSearchCV(
        estimator=ridge,
        param_distributions={"model__alpha": np.logspace(-3, 3, 30)},
        n_iter=10,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    ridge_search.fit(X_train, y_train, groups=g_train)
    print(f"Mejor R2 CV Ridge: {ridge_search.best_score_:.4f}")
    print(f"Mejor parámetro Ridge: {ridge_search.best_params_}")
    print()

    tuned_models = {
        "Regresion Lineal": linear,
        "Ridge optimizado": ridge_search.best_estimator_,
        "Random Forest optimizado": rf_search.best_estimator_,
        "Gradient Boosting optimizado": gb_search.best_estimator_,
    }

    results = []
    cv_fold_rows = []
    saved_predictions = test_df[["id", GROUP_COL, TARGET]].copy()

    for name, model in tuned_models.items():
        # Linear model was not searched; fit and evaluate with the same group CV.
        if name == "Regresion Lineal":
            model.fit(X_train, y_train)
            cv_scores = []
            for fold, (tr, va) in enumerate(cv.split(X_train, y_train, groups=g_train), start=1):
                fold_model = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression())
                ])
                fold_model.fit(X_train.iloc[tr], y_train.iloc[tr])
                pred_va = fold_model.predict(X_train.iloc[va])
                cv_scores.append({
                    "fold": fold,
                    "R2": r2_score(y_train.iloc[va], pred_va),
                    "MAE_MPa": mean_absolute_error(y_train.iloc[va], pred_va),
                    "RMSE_MPa": np.sqrt(mean_squared_error(y_train.iloc[va], pred_va)),
                })
            cv_df = pd.DataFrame(cv_scores)
        else:
            model.fit(X_train, y_train)
            rows = []
            for fold, (tr, va) in enumerate(cv.split(X_train, y_train, groups=g_train), start=1):
                fold_model = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
                # Use the already tuned pipeline parameters and refit per fold.
                fold_model.fit(X_train.iloc[tr], y_train.iloc[tr])
                pred_va = fold_model.predict(X_train.iloc[va])
                rows.append({
                    "fold": fold,
                    "R2": r2_score(y_train.iloc[va], pred_va),
                    "MAE_MPa": mean_absolute_error(y_train.iloc[va], pred_va),
                    "RMSE_MPa": np.sqrt(mean_squared_error(y_train.iloc[va], pred_va)),
                })
            cv_df = pd.DataFrame(rows)

        pred = model.predict(X_test)
        saved_predictions[f"pred_{name.lower().replace(' ', '_')}_mpa"] = pred
        test_r2 = r2_score(y_test, pred)
        test_mae = mean_absolute_error(y_test, pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "modelo": name,
            "R2_test": test_r2,
            "MAE_test_MPa": test_mae,
            "RMSE_test_MPa": test_rmse,
            "R2_CV_media": cv_df["R2"].mean(),
            "R2_CV_std": cv_df["R2"].std(ddof=0),
            "MAE_CV_media_MPa": cv_df["MAE_MPa"].mean(),
            "RMSE_CV_media_MPa": cv_df["RMSE_MPa"].mean(),
        })

        fold_tmp = cv_df.copy()
        fold_tmp.insert(1, "modelo", name)
        cv_fold_rows.append(fold_tmp)

        print(f"{name}: Test R2={test_r2:.4f} | MAE={test_mae:.4f} | RMSE={test_rmse:.4f} | CV R2={cv_df['R2'].mean():.4f} ± {cv_df['R2'].std(ddof=0):.4f}")

    results_df = pd.DataFrame(results).sort_values(
        ["R2_CV_media", "RMSE_CV_media_MPa"], ascending=[False, True]
    ).reset_index(drop=True)
    best_name = results_df.loc[0, "modelo"]
    best_model = tuned_models[best_name]

    print("\n=== MODELO SELECCIONADO ===")
    print(best_name)

    # Guardar métricas, CV por fold y predicciones.
    results_df.to_csv(RESULTS_DIR / "metricas_optimizadas.csv", index=False, encoding="utf-8-sig")
    pd.concat(cv_fold_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "cv_optimizacion_por_fold.csv", index=False, encoding="utf-8-sig"
    )
    saved_predictions.to_csv(RESULTS_DIR / "predicciones_test_optimizadas.csv", index=False, encoding="utf-8-sig")

    # Reentrenar el mejor modelo con los 445 registros individuales para el software.
    best_model.fit(X, y)
    joblib.dump(best_model, MODELS_DIR / "mejor_modelo_optimizado.joblib")

    # Importancia/perfil del modelo cuando sea posible.
    if hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("model"), "feature_importances_"):
        imp = pd.DataFrame({"variable": FEATURES, "importancia": best_model.named_steps["model"].feature_importances_})
        imp = imp.sort_values("importancia", ascending=False)
        imp.to_csv(RESULTS_DIR / "importancia_variables_mejor_modelo.csv", index=False, encoding="utf-8-sig")
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("model"), "coef_"):
        coef = pd.DataFrame({"variable": FEATURES, "coeficiente": best_model.named_steps["model"].coef_})
        coef["abs_coeficiente"] = coef["coeficiente"].abs()
        coef = coef.sort_values("abs_coeficiente", ascending=False)
        coef.to_csv(RESULTS_DIR / "coeficientes_mejor_modelo.csv", index=False, encoding="utf-8-sig")

    hiper = {
        "random_state": RANDOM_STATE,
        "split": {"method": "GroupShuffleSplit", "test_size": TEST_SIZE, "group_column": GROUP_COL},
        "cross_validation": {"method": "GroupKFold", "n_splits": N_SPLITS},
        "selection": "mayor R2_CV_media; desempate por menor RMSE_CV_media",
        "rf_search": rf_search.best_params_,
        "gb_search": gb_search.best_params_,
        "ridge_search": ridge_search.best_params_,
        "best_model": best_name,
    }
    with open(RESULTS_DIR / "hiperparametros_optimizados.json", "w", encoding="utf-8") as f:
        json.dump(hiper, f, ensure_ascii=False, indent=2)

    print("\n=== RESULTADOS ===")
    print(results_df.to_string(index=False))
    print(f"\nArchivos guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
