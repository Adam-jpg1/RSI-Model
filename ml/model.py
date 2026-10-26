import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from config import CONFIG
from ml.features import CLASS_LABELS

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Run: pip install xgboost")


class RSIModel:
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = RobustScaler()
        self.feature_names: list = []
        self.is_trained = False

    def build(self):
        params = {**CONFIG.ml.xgb_params}
        params.pop("use_label_encoder", None)
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        self.feature_names = list(X_train.columns)
        X_tr = self.scaler.fit_transform(X_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = self.scaler.transform(X_val)
            eval_set = [(X_v, y_val)]

        self.model.fit(
            X_tr, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_trained = True

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X[self.feature_names])
        probs = self.model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    def predict_single(self, feature_row: pd.Series) -> Dict:
        X = pd.DataFrame([feature_row[self.feature_names]])
        preds, probs = self.predict(X)
        label_idx = int(preds[0])
        return {
            "signal": CLASS_LABELS[label_idx],
            "confidence": float(probs[0][label_idx]),
            "probs": {CLASS_LABELS[i]: float(p) for i, p in enumerate(probs[0])},
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        preds, probs = self.predict(X_test)
        report = classification_report(
            y_test, preds,
            target_names=[CLASS_LABELS[i] for i in range(3)],
            output_dict=True,
        )
        cm = confusion_matrix(y_test, preds)
        acc = float(np.mean(preds == y_test))
        return {"accuracy": round(acc, 4), "report": report, "confusion_matrix": cm.tolist()}

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        scores = self.model.feature_importances_
        df = pd.DataFrame({"feature": self.feature_names, "importance": scores})
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    def shap_values(self, X: pd.DataFrame):
        try:
            import shap
        except ImportError:
            raise ImportError("Run: pip install shap")
        X_scaled = self.scaler.transform(X[self.feature_names])
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X_scaled)

    def save(self):
        os.makedirs(os.path.dirname(CONFIG.ml.model_path) or ".", exist_ok=True)
        self.model.save_model(CONFIG.ml.model_path)
        with open(CONFIG.ml.scaler_path, "wb") as f:
            pickle.dump((self.scaler, self.feature_names), f)
        print(f"Model saved to {CONFIG.ml.model_path}")

    def load(self):
        if not os.path.exists(CONFIG.ml.model_path):
            raise FileNotFoundError(f"No model at {CONFIG.ml.model_path}")
        self.build()
        self.model.load_model(CONFIG.ml.model_path)
        with open(CONFIG.ml.scaler_path, "rb") as f:
            self.scaler, self.feature_names = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {CONFIG.ml.model_path}")
