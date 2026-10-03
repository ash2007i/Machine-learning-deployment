```python
from flask import Flask, request, jsonify
import joblib
import threading
import time


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/content/math_difficulty_model.joblib"
PACKAGED_MODEL_PATH = "/content/scalar_math_difficulty_model.joblib"
TRAINING_LOG_PATH = "/content/training_logs.txt"

HOST = "0.0.0.0"
PORT = 5004


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

try:
    base_pipeline = joblib.load(MODEL_PATH)
    print("Original math difficulty model loaded successfully.")

except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    base_pipeline = None


# ============================================================
# PACKAGE MODEL WITH METADATA
# ============================================================

if base_pipeline is not None:

    scalar_model_package = {
        "model": base_pipeline,
        "version": 1.0,
        "accuracy_scalar": 0.60,
        "num_features_scalar": 1000
    }

    try:
        joblib.dump(
            scalar_model_package,
            PACKAGED_MODEL_PATH
        )

        print(
            "Packaged model saved successfully at:"
            f" {PACKAGED_MODEL_PATH}"
        )

    except Exception as e:
        print(f"ERROR: Could not save packaged model: {e}")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # Get JSON request
        data = request.get_json(force=True)

        # Get query
        query = data.get("query", "")

        # Validate query
        if not query or not isinstance(query, str):

            return jsonify({
                "error": 'Please provide a valid math "query".'
            }), 400

        # Load packaged model
        pkg = joblib.load(PACKAGED_MODEL_PATH)

        # Extract model
        model = pkg["model"]

        # Make prediction
        prediction = model.predict([query])[0]

        # Convert prediction safely to string
        prediction_text = str(prediction).upper()

        # Return response
        return jsonify({

            "query": query,

            "predicted_difficulty": prediction_text,

            "model_version": pkg["version"],

            "model_accuracy": pkg["accuracy_scalar"]

        }), 200

    except Exception as e:

        return jsonify({

            "error": f"Prediction failed: {str(e)}"

        }), 500


# ============================================================
# TRAINING DATA API
# ============================================================

@app.route("/api/train", methods=["POST"])
def train_append():

    """
    Register a new query-difficulty mapping.

    IMPORTANT:
    This endpoint does NOT actually retrain the ML model.

    It only stores new training examples so that the model
    can be retrained later.
    """

    try:

        # Get JSON request
        data = request.get_json(force=True)

        # Extract values
        query = data.get("query", "")
        difficulty = data.get("difficulty", "")

        # Validate input
        if not query or not difficulty:

            return jsonify({

                "error": (
                    'Please provide both "query" and '
                    '"difficulty" to register training.'
                )

            }), 400

        # Append training data to log
        with open(
            TRAINING_LOG_PATH,
            "a",
            encoding="utf-8"
        ) as log_file:

            log_file.write(
                f"{query}|||{difficulty}\n"
            )

        # Return success
        return jsonify({

            "status": "success",

            "message": (
                "New pattern recorded. "
                "Ready for retraining iteration."
            )

        }), 200

    except Exception as e:

        return jsonify({

            "error": f"Registration failed: {str(e)}"

        }), 500


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "online",

        "service": "Scalar Math Difficulty Model API",

        "port": PORT,

        "model_loaded": base_pipeline is not None

    }), 200


# ============================================================
# FLASK SERVER FUNCTION
# ============================================================

def run_api():

    app.run(

        host=HOST,

        port=PORT,

        debug=False,

        use_reloader=False

    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print("   SCALAR MATH DIFFICULTY MODEL API")
    print("==============================================")
    print(f"Server: http://127.0.0.1:{PORT}")
    print("")
    print("Available endpoints:")
    print(f"  POST http://127.0.0.1:{PORT}/api/predict")
    print(f"  POST http://127.0.0.1:{PORT}/api/train")
    print(f"  GET  http://127.0.0.1:{PORT}/api/health")
    print("==============================================")
    print("")

    # Start Flask
    run_api()
```
