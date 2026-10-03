from flask import Flask, request, jsonify
import joblib
import os


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "math_difficulty_model.joblib"
)

PACKAGED_MODEL_PATH = os.path.join(
    BASE_DIR,
    "scalar_math_difficulty_model.joblib"
)

TRAINING_LOG_PATH = os.path.join(
    BASE_DIR,
    "training_logs.txt"
)

HOST = "0.0.0.0"

# Render provides the PORT environment variable
PORT = int(os.environ.get("PORT", 5004))


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

        print(
            f"ERROR: Could not save packaged model: {e}"
        )


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(force=True)

        query = data.get("query", "")

        if not query or not isinstance(query, str):

            return jsonify({
                "error": 'Please provide a valid math "query".'
            }), 400

        # Make sure model exists
        if base_pipeline is None:

            return jsonify({
                "error": "Model is not loaded on the server."
            }), 500

        # Load packaged model
        pkg = joblib.load(PACKAGED_MODEL_PATH)

        model = pkg["model"]

        # Predict
        prediction = model.predict([query])[0]

        prediction_text = str(prediction).upper()

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
    Records new query/difficulty pairs.

    IMPORTANT:
    This does NOT retrain the ML model.
    """

    try:

        data = request.get_json(force=True)

        query = data.get("query", "")

        difficulty = data.get("difficulty", "")

        if not query or not difficulty:

            return jsonify({

                "error": (
                    'Please provide both "query" and '
                    '"difficulty".'
                )

            }), 400

        with open(
            TRAINING_LOG_PATH,
            "a",
            encoding="utf-8"
        ) as log_file:

            log_file.write(
                f"{query}|||{difficulty}\n"
            )

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
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "online",

        "service": "Scalar Math Difficulty Model API",

        "model_loaded": base_pipeline is not None

    }), 200


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "service": "Scalar Math Difficulty Model API",

        "status": "online",

        "endpoints": {

            "health": "GET /api/health",

            "predict": "POST /api/predict",

            "train": "POST /api/train"

        }

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("==============================================")
    print("   SCALAR MATH DIFFICULTY MODEL API")
    print("==============================================")

    print(f"Starting server on port {PORT}")

    print("==============================================")

    app.run(
        host=HOST,
        port=PORT,
        debug=False
    )
