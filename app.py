from flask import Flask, request, jsonify
import joblib
import os
import re
import ast
import operator


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
PORT = int(os.environ.get("PORT", 5004))


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    base_pipeline = joblib.load(MODEL_PATH)

    print("Original math difficulty model loaded successfully.")

except Exception as e:

    print(f"ERROR: Could not load model: {e}")

    base_pipeline = None


# ============================================================
# PACKAGE MODEL
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
            f"Packaged model saved successfully at: "
            f"{PACKAGED_MODEL_PATH}"
        )

    except Exception as e:

        print(f"ERROR: Could not save packaged model: {e}")


# ============================================================
# SAFE MATH SOLVER
# ============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod
}


def safe_calculate(expression):

    """
    Safely evaluates basic mathematical expressions.

    Supports:
        +  -  *  /  %  **
        parentheses
        positive/negative numbers
    """

    expression = expression.strip()

    # Convert common math symbols
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("^", "**")

    # Remove common question wording
    expression = re.sub(
        r"(?i)(what is|calculate|solve|find|evaluate)\s+",
        "",
        expression
    )

    expression = expression.rstrip("?").strip()

    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

        def evaluate(node):

            if isinstance(node, ast.Expression):

                return evaluate(node.body)

            if isinstance(node, ast.Constant):

                if isinstance(node.value, (int, float)):

                    return node.value

                raise ValueError("Invalid constant")

            if isinstance(node, ast.UnaryOp):

                value = evaluate(node.operand)

                if isinstance(node.op, ast.USub):
                    return -value

                if isinstance(node.op, ast.UAdd):
                    return value

                raise ValueError("Invalid unary operator")

            if isinstance(node, ast.BinOp):

                left = evaluate(node.left)
                right = evaluate(node.right)

                operation = OPERATORS.get(
                    type(node.op)
                )

                if operation is None:
                    raise ValueError("Operator not supported")

                return operation(left, right)

            raise ValueError(
                "Expression contains unsupported elements"
            )

        result = evaluate(tree)

        return result

    except Exception:

        return None


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

        if base_pipeline is None:

            return jsonify({
                "error": "Model is not loaded on the server."
            }), 500

        pkg = joblib.load(
            PACKAGED_MODEL_PATH
        )

        model = pkg["model"]

        prediction = model.predict([query])[0]

        prediction_text = str(
            prediction
        ).upper()

        response = {

            "query": query,

            "predicted_difficulty":
                prediction_text,

            "model_version":
                pkg["version"],

            "model_accuracy":
                pkg["accuracy_scalar"]

        }

        # Add confidence if the model supports it
        try:

            probabilities = model.predict_proba(
                [query]
            )[0]

            classes = model.classes_

            confidence = max(probabilities)

            response["prediction_confidence"] = float(
                confidence
            )

            response["class_probabilities"] = {
                str(classes[i]):
                float(probabilities[i])
                for i in range(len(classes))
            }

        except Exception:

            response["prediction_confidence"] = None

        return jsonify(response), 200

    except Exception as e:

        return jsonify({

            "error":
                f"Prediction failed: {str(e)}"

        }), 500


# ============================================================
# SOLVE API
# ============================================================

@app.route("/api/solve", methods=["POST"])
def solve():

    try:

        data = request.get_json(force=True)

        query = data.get("query", "")

        if not query or not isinstance(query, str):

            return jsonify({

                "error":
                    'Please provide a valid math "query".'

            }), 400

        # ----------------------------------------------------
        # 1. GET ML MODEL PREDICTION
        # ----------------------------------------------------

        if base_pipeline is None:

            return jsonify({

                "error":
                    "Model is not loaded on the server."

            }), 500

        pkg = joblib.load(
            PACKAGED_MODEL_PATH
        )

        model = pkg["model"]

        prediction = model.predict([query])[0]

        difficulty = str(
            prediction
        ).upper()

        # ----------------------------------------------------
        # 2. SOLVE THE MATHEMATICAL EXPRESSION
        # ----------------------------------------------------

        answer = safe_calculate(query)

        # ----------------------------------------------------
        # 3. BUILD RESPONSE
        # ----------------------------------------------------

        if answer is None:

            solution_status = "UNABLE_TO_SOLVE"

            answer_text = (
                "I could not solve this expression "
                "with the current mathematical solver."
            )

        else:

            solution_status = "SOLVED"

            if isinstance(answer, float) and answer.is_integer():

                answer = int(answer)

            answer_text = str(answer)

        response = {

            "query": query,

            "answer": answer_text,

            "solution_status":
                solution_status,

            "predicted_difficulty":
                difficulty,

            "model_accuracy":
                pkg["accuracy_scalar"],

            "model_version":
                pkg["version"]

        }

        # ----------------------------------------------------
        # 4. MODEL CONFIDENCE
        # ----------------------------------------------------

        try:

            probabilities = model.predict_proba(
                [query]
            )[0]

            classes = model.classes_

            confidence = max(probabilities)

            response["prediction_confidence"] = float(
                confidence
            )

            response["class_probabilities"] = {

                str(classes[i]):
                float(probabilities[i])

                for i in range(len(classes))

            }

        except Exception:

            response["prediction_confidence"] = None

        # ----------------------------------------------------
        # 5. ANSWER ACCURACY
        # ----------------------------------------------------
        #
        # We cannot honestly calculate answer accuracy
        # without a known correct answer.
        #
        # If the client sends:
        #
        # {
        #     "query": "2 + 3",
        #     "expected_answer": 5
        # }
        #
        # we can compare the result.
        # ----------------------------------------------------

        expected_answer = data.get(
            "expected_answer",
            None
        )

        if expected_answer is not None and answer is not None:

            try:

                expected_numeric = float(
                    expected_answer
                )

                answer_numeric = float(
                    answer
                )

                if abs(
                    expected_numeric -
                    answer_numeric
                ) < 1e-9:

                    response["answer_accuracy"] = 1.0

                else:

                    response["answer_accuracy"] = 0.0

            except Exception:

                response["answer_accuracy"] = None

        else:

            response["answer_accuracy"] = None

        return jsonify(response), 200

    except Exception as e:

        return jsonify({

            "error":
                f"Solving failed: {str(e)}"

        }), 500


# ============================================================
# TRAINING DATA API
# ============================================================

@app.route("/api/train", methods=["POST"])
def train_append():

    try:

        data = request.get_json(force=True)

        query = data.get("query", "")

        difficulty = data.get(
            "difficulty",
            ""
        )

        if not query or not difficulty:

            return jsonify({

                "error":
                    'Please provide both "query" and '
                    '"difficulty".'

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

            "status":
                "success",

            "message":
                "New pattern recorded. "
                "Ready for retraining iteration."

        }), 200

    except Exception as e:

        return jsonify({

            "error":
                f"Registration failed: {str(e)}"

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status":
            "online",

        "service":
            "Scalar Math Difficulty Model API",

        "model_loaded":
            base_pipeline is not None

    }), 200


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "service":
            "Scalar Math Difficulty Model API",

        "status":
            "online",

        "endpoints": {

            "health":
                "GET /api/health",

            "predict":
                "POST /api/predict",

            "solve":
                "POST /api/solve",

            "train":
                "POST /api/train"

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
