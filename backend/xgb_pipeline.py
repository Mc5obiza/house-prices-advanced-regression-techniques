from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor

DROP_COLUMNS = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
TRAIN_DROPNA_SUBSET = ["MasVnrArea", "Electrical"]

SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "xgb_house_price.joblib"
LEGACY_MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "xgb_house_price.pkl"

BEST_XGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 1.0,
    "gamma": 0,
    "reg_alpha": 0.2,
    "reg_lambda": 2,
    "n_estimators": 200,
}

FEATURE_IMPORTANCE_THRESHOLD = "median"


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Replicates the feature engineering logic from the notebook."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HouseFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["TotalBathrooms"] = (
            self._num(df, "FullBath")
            + 0.5 * self._num(df, "HalfBath")
            + self._num(df, "BsmtFullBath")
            + 0.5 * self._num(df, "BsmtHalfBath")
        )

        df["TotalPorchSF"] = (
            self._num(df, "OpenPorchSF")
            + self._num(df, "EnclosedPorch")
            + self._num(df, "3SsnPorch")
            + self._num(df, "ScreenPorch")
        )

        df["HouseAge"] = self._num(df, "YrSold") - self._num(df, "YearBuilt")
        df["RemodAge"] = self._num(df, "YrSold") - self._num(df, "YearRemodAdd")
        df["SinceGarageBuilt"] = self._num(df, "YrSold") - self._num(df, "GarageYrBlt")
        df["TotalSF"] = (
            self._num(df, "TotalBsmtSF") + self._num(df, "1stFlrSF") + self._num(df, "2ndFlrSF")
        )

        df["OverallGrade"] = self._num(df, "OverallQual") * self._num(df, "OverallCond")
        df["TotalRooms"] = self._num(df, "TotRmsAbvGrd") + self._num(df, "KitchenAbvGr")

        rooms_non_zero = self._num(df, "TotRmsAbvGrd").replace(0, 1)
        garage_cars_non_zero = self._num(df, "GarageCars").replace(0, 1)

        df["BedroomsPerRoom"] = self._num(df, "BedroomAbvGr") / rooms_non_zero
        df["GarageEfficiency"] = self._num(df, "GarageArea") / garage_cars_non_zero

        year_built = self._num(df, "YearBuilt")
        year_remod = self._num(df, "YearRemodAdd")
        df["IsRemodeled"] = (
            year_built.notna() & year_remod.notna() & (year_built != year_remod)
        ).astype(int)

        df["HasGarage"] = self._obj(df, "GarageType").notna().astype(int)
        df["HasPool"] = self._obj(df, "PoolQC").notna().astype(int)
        df["HasFence"] = self._obj(df, "Fence").notna().astype(int)
        df["HasFireplace"] = self._obj(df, "FireplaceQu").notna().astype(int)

        df["SeasonSold"] = self._num(df, "MoSold").map(SEASON_MAP)

        df["AgeCategory"] = pd.cut(
            df["HouseAge"],
            bins=[-1, 10, 50, 200],
            labels=["New", "Mid", "Old"],
        ).astype("object")

        df["QualitySFInteraction"] = self._num(df, "GrLivArea") * self._num(df, "OverallQual")
        df["BathsPerBedroom"] = df["TotalBathrooms"] / self._num(df, "BedroomAbvGr")
        df["PorchVsLotArea"] = df["TotalPorchSF"] / self._num(df, "LotArea")
        df["BasementRatio"] = self._num(df, "TotalBsmtSF") / df["TotalSF"]
        df["LivingToLotRatio"] = self._num(df, "GrLivArea") / self._num(df, "LotArea")

        df = df.drop(columns=DROP_COLUMNS, errors="ignore")
        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def _num(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(0.0, index=df.index, dtype="float64")

    @staticmethod
    def _obj(df: pd.DataFrame, col: str) -> pd.Series:
        if col in df.columns:
            return df[col].astype("object")
        return pd.Series(np.nan, index=df.index, dtype="object")


class NotebookStyleNumericImputer(BaseEstimator, TransformerMixin):
    """Applies the same selective mean/median numeric filling used in the notebook."""

    def __init__(self) -> None:
        self.fill_values_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NotebookStyleNumericImputer":
        self.fill_values_ = {}

        na_count = X.isna().sum()
        missing_cols = na_count[na_count > 0].index.tolist()
        numeric_missing_cols = [col for col in missing_cols if X[col].dtype != "object"]

        if numeric_missing_cols:
            first_col = numeric_missing_cols[0]
            self.fill_values_[first_col] = float(X[first_col].median())

            if len(numeric_missing_cols) > 1:
                second_col = numeric_missing_cols[1]
                self.fill_values_[second_col] = float(X[second_col].mean())

            if len(numeric_missing_cols) > 2:
                third_col = numeric_missing_cols[2]
                self.fill_values_[third_col] = float(X[third_col].mean())

            # Matches notebook ordering where the last numeric missing column is mean-filled.
            last_col = numeric_missing_cols[-1]
            self.fill_values_[last_col] = float(X[last_col].mean())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()
        for col, fill_value in self.fill_values_.items():
            if col in X_filled.columns:
                X_filled[col] = X_filled[col].fillna(fill_value)
        return X_filled


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _select_num_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=["object"]).columns.tolist()


def _select_cat_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object"]).columns.tolist()


def _build_xgb_estimator() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=18,
        n_jobs=-1,
        **BEST_XGB_PARAMS,
    )


def build_xgb_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, _select_num_columns),
            ("cat", categorical_pipeline, _select_cat_columns),
        ]
    )

    feature_selector = SelectFromModel(
        estimator=_build_xgb_estimator(),
        threshold=FEATURE_IMPORTANCE_THRESHOLD,
    )

    model = _build_xgb_estimator()

    return Pipeline(
        steps=[
            ("feature_engineering", HouseFeatureEngineer()),
            ("notebook_num_imputer", NotebookStyleNumericImputer()),
            ("preprocessor", preprocessor),
            ("feature_selector", feature_selector),
            ("model", model),
        ]
    )


def save_model(model: Pipeline, model_path: str | Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: str | Path) -> Pipeline:
    # Backward compatibility for models pickled when this file was run as __main__.
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "HouseFeatureEngineer", HouseFeatureEngineer)
        setattr(main_module, "NotebookStyleNumericImputer", NotebookStyleNumericImputer)
        setattr(main_module, "_select_num_columns", _select_num_columns)
        setattr(main_module, "_select_cat_columns", _select_cat_columns)

    target_model_path = Path(model_path)
    if (
        target_model_path == MODEL_OUTPUT_PATH
        and not target_model_path.exists()
        and LEGACY_MODEL_OUTPUT_PATH.exists()
    ):
        target_model_path = LEGACY_MODEL_OUTPUT_PATH

    loaded_model = joblib.load(target_model_path)

    # Auto-migrate legacy .pkl artifact to .joblib the first time it is loaded.
    if target_model_path == LEGACY_MODEL_OUTPUT_PATH and not MODEL_OUTPUT_PATH.exists():
        save_model(loaded_model, MODEL_OUTPUT_PATH)

    return loaded_model


def train_and_save_model() -> Path:
    data = load_dataset(TRAIN_CSV_PATH)

    existing_dropna_cols = [col for col in TRAIN_DROPNA_SUBSET if col in data.columns]
    if existing_dropna_cols:
        data = data.dropna(subset=existing_dropna_cols)

    if "SalePrice" not in data.columns:
        raise ValueError("Training CSV must contain a SalePrice column.")

    X = data.drop(columns=["SalePrice"])
    y = np.log1p(data["SalePrice"])

    pipeline = build_xgb_pipeline()
    pipeline.fit(X, y)
    save_model(pipeline, MODEL_OUTPUT_PATH)
    return MODEL_OUTPUT_PATH


def predict_from_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path | None = None,
    model_path: str | Path = MODEL_OUTPUT_PATH,
) -> pd.DataFrame:
    pipeline = load_model(model_path)
    data = load_dataset(input_csv_path)

    X = data.drop(columns=["SalePrice"], errors="ignore")
    pred_log = pipeline.predict(X)
    pred = np.expm1(pred_log)

    prediction_df = pd.DataFrame({"SalePrice": pred})
    if "Id" in data.columns:
        prediction_df.insert(0, "Id", data["Id"].tolist())

    if output_csv_path is not None:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(output_path, index=False)

    return prediction_df


if __name__ == "__main__":
    saved_model_path = train_and_save_model()
    print(f"Model trained on full dataset and saved to: {saved_model_path}")
