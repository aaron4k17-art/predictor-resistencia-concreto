from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURACION
# ============================================================
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
    "cemento_kg_m3",
    "escoria_kg_m3",
    "fly_ash_kg_m3",
    "agua_kg_m3",
    "superplastificante_kg_m3",
    "agregado_grueso_kg_m3",
    "agregado_fino_kg_m3",
    "edad_dias",
]

TARGET = "resistencia_mpa"
GROUP_COL = "mezcla_id"
NATURE_COL = "naturaleza_observacion"

# ============================================================
# 2. CARGA Y VALIDACION
# ============================================================
def cargar_datos(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    df = pd.read_excel(path)

    required = FEATURES + [TARGET, GROUP_COL, NATURE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Para el modelo principal solo se usan resultados individuales.
    df = df[df[NATURE_COL].astype(str).str.upper().eq("INDIVIDUAL")].copy()

    # Convertir variables numericas.
    numeric_cols = FEATURES + [TARGET]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar filas incompletas solo en las columnas que usa el modelo.
    before = len(df)
    df = df.dropna(subset=required).copy()
    after = len(df)

    # Eliminar duplicados exactos de los predictores + objetivo + mezcla.
    dup_cols = FEATURES + [TARGET, GROUP_COL]
    before_dup = len(df)
    df = df.drop_duplicates(subset=dup_cols).copy()
    duplicates_removed = before_dup - len(df)

    print("\n=== CONTROL DE DATOS ===")
    print(f"Registros individuales iniciales: {before}")
    print(f"Registros con datos completos:   {after}")
    print(f"Duplicados exactos eliminados:   {duplicates_removed}")
    print(f"Registros finales para ML:       {len(df)}")
    print(f"Mezclas/grupos unicos:            {df[GROUP_COL].nunique()}")

    if len(df) < 100:
        raise ValueError("La cantidad de datos individuales es demasiado baja para este entrenamiento.")

    return df


# ============================================================
# 3. MODELOS BASE
# ============================================================
def construir_modelos():
    # Escalado solamente para Regresion Lineal.
    linear = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    random_state=RANDOM_STATE,
                    loss="squared_error",
                ),
            ),
        ]
    )

    return {
        "Regresion Lineal": linear,
        "Random Forest": rf,
        "Gradient Boosting": gb,
    }


# ============================================================
# 4. DIVISION GROUP-AWARE 80/20
# ============================================================
def separar_train_test(df: pd.DataFrame):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    idx_train, idx_test = next(splitter.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))

    train_df = df.iloc[idx_train].copy()
    test_df = df.iloc[idx_test].copy()

    # Comprobacion de no fuga entre grupos.
    overlap = set(train_df[GROUP_COL]).intersection(set(test_df[GROUP_COL]))
    if overlap:
        raise RuntimeError(f"Fuga de grupos detectada: {overlap}")

    print("\n=== PARTICION TRAIN / TEST ===")
    print(f"Train: {len(train_df)} registros ({len(train_df)/len(df):.1%})")
    print(f"Test:  {len(test_df)} registros ({len(test_df)/len(df):.1%})")
    print(f"Grupos train: {train_df[GROUP_COL].nunique()}")
    print(f"Grupos test:  {test_df[GROUP_COL].nunique()}")
    print("No hay mezcla_id compartidos entre train y test.")

    return train_df, test_df


# ============================================================
# 5. ENTRENAMIENTO + METRICAS DE TEST + 5-FOLD GROUP CV
# ============================================================
def evaluar_modelos(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    groups_train = train_df[GROUP_COL]

    models = construir_modelos()
    cv = GroupKFold(n_splits=N_SPLITS)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    resultados = []
    predicciones = {}

    for name, model in models.items():
        print(f"\nEntrenando: {name}")
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        predicciones[name] = pred

        test_r2 = r2_score(y_test, pred)
        test_mae = mean_absolute_error(y_test, pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, pred))

        cv_result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            groups=groups_train,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        cv_r2 = cv_result["test_r2"]
        cv_mae = -cv_result["test_mae"]
        cv_rmse = -cv_result["test_rmse"]

        row = {
            "modelo": name,
            "R2_test": test_r2,
            "MAE_test_MPa": test_mae,
            "RMSE_test_MPa": test_rmse,
            "R2_CV_media": cv_r2.mean(),
            "R2_CV_std": cv_r2.std(),
            "MAE_CV_media_MPa": cv_mae.mean(),
            "RMSE_CV_media_MPa": cv_rmse.mean(),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_grupos_train": train_df[GROUP_COL].nunique(),
            "n_grupos_test": test_df[GROUP_COL].nunique(),
        }
        resultados.append(row)

        print(
            f"  Test -> R2={test_r2:.4f} | MAE={test_mae:.4f} MPa | RMSE={test_rmse:.4f} MPa"
        )
        print(
            f"  CV   -> R2={cv_r2.mean():.4f} +/- {cv_r2.std():.4f} | "
            f"MAE={cv_mae.mean():.4f} MPa | RMSE={cv_rmse.mean():.4f} MPa"
        )

    results_df = pd.DataFrame(resultados)

    # Criterio principal: mayor R2 de CV; desempate por menor RMSE CV.
    results_df = results_df.sort_values(
        by=["R2_CV_media", "RMSE_CV_media_MPa"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best_name = results_df.loc[0, "modelo"]
    best_model = models[best_name]

    print("\n=== MEJOR MODELO ===")
    print(best_name)

    # Reentrenar el mejor modelo usando TODOS los datos individuales.
    X_all = df_global[FEATURES]
    y_all = df_global[TARGET]
    best_model.fit(X_all, y_all)
    joblib.dump(best_model, MODELS_DIR / "mejor_modelo.joblib")

    # Guardar todos los modelos entrenados con train.
    for name, model in models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.joblib")

    # Predicciones del conjunto de prueba para comparacion.
    pred_test = test_df[["id", GROUP_COL, TARGET]].copy()
    for name, pred in predicciones.items():
        pred_test[f"pred_{name.lower().replace(' ', '_')}_mpa"] = pred
    pred_test.to_csv(RESULTS_DIR / "predicciones_test.csv", index=False, encoding="utf-8-sig")

    results_df.to_csv(RESULTS_DIR / "metricas_modelos.csv", index=False, encoding="utf-8-sig")

    # CV por pliegue del mejor modelo para mayor trazabilidad.
    cv_best = cross_validate(
        models[best_name],
        X_train,
        y_train,
        cv=cv,
        groups=groups_train,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    cv_df = pd.DataFrame({
        "fold": np.arange(1, N_SPLITS + 1),
        "modelo": best_name,
        "R2": cv_best["test_r2"],
        "MAE_MPa": -cv_best["test_mae"],
        "RMSE_MPa": -cv_best["test_rmse"],
    })
    cv_df.to_csv(RESULTS_DIR / "cv_mejor_modelo.csv", index=False, encoding="utf-8-sig")

    # Hiperparametros usados.
    hiper = {
        "random_state": RANDOM_STATE,
        "split": {
            "method": "GroupShuffleSplit",
            "test_size": TEST_SIZE,
        },
        "cross_validation": {
            "method": "GroupKFold",
            "n_splits": N_SPLITS,
            "group_column": GROUP_COL,
        },
        "models": {
            "Regresion Lineal": {
                "fit_intercept": True,
                "preprocessing": "SimpleImputer(median) + StandardScaler",
            },
            "Random Forest": {
                "n_estimators": 500,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "Gradient Boosting": {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 3,
                "loss": "squared_error",
                "random_state": RANDOM_STATE,
            },
        },
    }
    with open(RESULTS_DIR / "hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(hiper, f, ensure_ascii=False, indent=2)

    return results_df, best_name


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    df_global = cargar_datos(DATA_PATH)
    train_df, test_df = separar_train_test(df_global)
    resultados, mejor = evaluar_modelos(train_df, test_df)

    print("\n=== RESULTADOS FINALES ===")
    print(resultados.to_string(index=False))
    print(f"\nModelo seleccionado por CV: {mejor}")
    print(f"Resultados guardados en: {RESULTS_DIR}")
    print(f"Modelos guardados en:    {MODELS_DIR}")
