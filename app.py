import hmac
import importlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model_lstm_metar.h5"
SCALER_PATH = PROJECT_ROOT / "lstm_scaler.pkl"
FEATURE_ORDER = ["suhu_c", "qnh_hpa", "kec_angin_kt", "dew_point_c"]
TIME_STEPS = 10
_model = None
_scaler = None


def load_lstm_assets():
    global _model, _scaler
    if _model is None or _scaler is None:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        keras_models = importlib.import_module("keras.models")
        joblib = importlib.import_module("joblib")
        _model = keras_models.load_model(str(MODEL_PATH), compile=False)
        _scaler = joblib.load(SCALER_PATH)

        input_shape = _model.input_shape
        output_shape = _model.output_shape
        if len(input_shape) != 3 or input_shape[1:] != (TIME_STEPS, len(FEATURE_ORDER)):
            raise ValueError(f"Bentuk input model tidak sesuai: {input_shape}")
        if len(output_shape) != 2 or output_shape[-1] != len(FEATURE_ORDER):
            raise ValueError(f"Bentuk output model tidak sesuai: {output_shape}")
        scaler_features = list(getattr(_scaler, "feature_names_in_", FEATURE_ORDER))
        if scaler_features != FEATURE_ORDER:
            raise ValueError(f"Urutan fitur scaler tidak sesuai: {scaler_features}")
        if getattr(_scaler, "n_features_in_", len(FEATURE_ORDER)) != len(FEATURE_ORDER):
            raise ValueError("Scaler harus memiliki tepat 4 fitur")
    return _model, _scaler


def authorized_request():
    expected_token = os.environ.get("LSTM_SERVICE_TOKEN", "")
    supplied_token = request.headers.get("Authorization", "")
    if not expected_token:
        return False
    return hmac.compare_digest(supplied_token, f"Bearer {expected_token}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "metar-lstm-inference"})


@app.route("/predict", methods=["POST"])
def predict_lstm():
    if not authorized_request():
        return jsonify({"error": "Inference service is not configured or unauthorized."}), 401

    payload = request.get_json(silent=True)
    sequence = payload.get("sequence") if isinstance(payload, dict) else None
    if not isinstance(sequence, list) or len(sequence) != TIME_STEPS:
        return jsonify({"error": f"sequence harus berisi {TIME_STEPS} time-step."}), 400

    try:
        import numpy as np

        input_values = np.asarray(sequence, dtype=np.float32)
        if input_values.shape != (TIME_STEPS, len(FEATURE_ORDER)):
            return jsonify({"error": "Setiap time-step harus memiliki 4 fitur dalam urutan yang ditentukan."}), 400
        if not np.isfinite(input_values).all():
            return jsonify({"error": "Sequence memiliki nilai yang tidak valid."}), 400

        model, scaler = load_lstm_assets()
        scaled_input = scaler.transform(input_values)
        prediction_scaled = model.predict(scaled_input.reshape(1, TIME_STEPS, len(FEATURE_ORDER)), verbose=0)
        prediction_scaled = np.asarray(prediction_scaled, dtype=np.float32)
        if prediction_scaled.shape != (1, len(FEATURE_ORDER)):
            raise ValueError(f"Bentuk output prediksi tidak sesuai: {prediction_scaled.shape}")
        forecast = scaler.inverse_transform(prediction_scaled)[0]
        if not np.isfinite(forecast).all():
            raise ValueError("Model menghasilkan nilai prediksi yang tidak valid")

        reference_time = pd_to_datetime(payload.get("latest_time"))
        forecast_time = (reference_time + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        predicted = {name: round(float(value), 3) for name, value in zip(FEATURE_ORDER, forecast)}
        actual = {name: round(float(value), 3) for name, value in zip(FEATURE_ORDER, input_values[-1])}
        return jsonify({
            "forecast_horizon_hours": 1,
            "latest_time": reference_time.isoformat().replace("+00:00", "Z"),
            "forecast_time": forecast_time,
            "features": FEATURE_ORDER,
            "actual": actual,
            "predicted": predicted,
            "history": payload.get("history", []),
        })
    except Exception as error:
        app.logger.exception("LSTM forecast failed")
        return jsonify({"error": "Prediksi LSTM gagal diproses.", "error_code": type(error).__name__}), 503


def pd_to_datetime(value):
    from datetime import datetime, timezone

    if not value:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
