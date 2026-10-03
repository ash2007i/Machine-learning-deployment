from flask import Flask, request, jsonify
import joblib
import os
import re
import sympy as sp

# ============================================================
# CONFIG
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

app = Flask(__name__)


# ============================================================
# LOAD DIFFICULTY MODEL
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
            "Packaged model saved successfully at: "
            f"{PACKAGED_MODEL_PATH}"
        )

    except Exception as e:
        print(f"ERROR: Could not save packaged model: {e}")


# ============================================================
# DIFFICULTY PREDICTION
# ============================================================

def predict_difficulty(query):

    if base_pipeline is None:
        return {
            "difficulty": None,
            "confidence": None,
            "probabilities": {}
        }

    prediction = base_pipeline.predict([query])[0]

    probabilities = {}

    confidence = None

    try:

        if hasattr(base_pipeline, "predict_proba"):

            probs = base_pipeline.predict_proba([query])[0]

            classes = base_pipeline.classes_

            for cls, prob in zip(classes, probs):
                probabilities[str(cls)] = round(
                    float(prob),
                    4
                )

            confidence = round(
                float(max(probs)),
                4
            )

    except Exception as e:

        print(
            f"Probability calculation failed: {e}"
        )

    return {
        "difficulty": str(prediction).upper(),
        "confidence": confidence,
        "probabilities": probabilities
    }


# ============================================================
# CLEAN MATHEMATICAL INPUT
# ============================================================

def clean_expression(text):

    expression = text.strip()

    # Unicode replacements
    expression = expression.replace("²", "^2")
    expression = expression.replace("³", "^3")
    expression = expression.replace("⁴", "^4")
    expression = expression.replace("⁵", "^5")

    expression = expression.replace("−", "-")
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")

    # Convert ^ to **
    expression = expression.replace("^", "**")

    return expression


# ============================================================
# SOLVE EQUATIONS
# ============================================================

def solve_equation(query):

    x = sp.symbols("x")

    expression = clean_expression(query)

    # Remove common natural-language prefixes
    prefixes = [
        "solve",
        "solve for x",
        "find x",
        "calculate x"
    ]

    lower_expression = expression.lower()

    for prefix in prefixes:

        if lower_expression.startswith(prefix):

            expression = expression[
                len(prefix):
            ].strip()

            break

    # Remove trailing punctuation
    expression = expression.rstrip("?.!")

    # Equation
    if "=" in expression:

        left, right = expression.split("=", 1)

        left_expr = sp.sympify(left)
        right_expr = sp.sympify(right)

        equation = sp.Eq(
            left_expr,
            right_expr
        )

        solutions = sp.solve(
            equation,
            x
        )

    else:

        expr = sp.sympify(expression)

        solutions = sp.solve(
            expr,
            x
        )

    if not solutions:

        return None

    steps = []

    steps.append(
        f"Original problem: {query}"
    )

    if "=" in expression:

        steps.append(
            f"Equation: {expression}"
        )

    steps.append(
        f"Solutions: {', '.join(map(str, solutions))}"
    )

    answer = ", ".join(
        f"x = {solution}"
        for solution in solutions
    )

    return {
        "answer": answer,
        "steps": steps
    }


# ============================================================
# DERIVATIVE
# ============================================================

def solve_derivative(query):

    x = sp.symbols("x")

    expression = query

    lower = expression.lower()

    keywords = [
        "derivative of",
        "differentiate",
        "differentiate"
    ]

    for keyword in keywords:

        if keyword in lower:

            index = lower.find(keyword)

            expression = expression[
                index + len(keyword):
            ].strip()

            break

    expression = clean_expression(expression)

    expression = expression.rstrip("?.!")

    expr = sp.sympify(expression)

    result = sp.diff(
        expr,
        x
    )

    steps = [
        f"Function: {expr}",
        "Differentiate with respect to x.",
        f"Derivative: {result}"
    ]

    return {
        "answer": str(result),
        "steps": steps
    }


# ============================================================
# INTEGRAL
# ============================================================

def solve_integral(query):

    x = sp.symbols("x")

    expression = query

    lower = expression.lower()

    # Definite integral:
    # "integral of x^2 from 0 to 5"

    pattern = r"integral of (.+?) from (.+?) to (.+)"

    match = re.search(
        pattern,
        lower
    )

    if match:

        function_text = match.group(1)
        lower_bound = match.group(2)
        upper_bound = match.group(3)

        function_text = clean_expression(
            function_text
        )

        function = sp.sympify(
            function_text
        )

        a = sp.sympify(
            clean_expression(lower_bound)
        )

        b = sp.sympify(
            clean_expression(upper_bound)
        )

        result = sp.integrate(
            function,
            (x, a, b)
        )

        steps = [
            f"Function: {function}",
            f"Bounds: {a} to {b}",
            f"Integral: {sp.integrate(function, x)}",
            f"Final value: {result}"
        ]

        return {
            "answer": str(result),
            "steps": steps
        }

    # Indefinite integral

    keywords = [
        "integral of",
        "integrate"
    ]

    for keyword in keywords:

        if keyword in lower:

            index = lower.find(keyword)

            expression = expression[
                index + len(keyword):
            ].strip()

            break

    expression = clean_expression(
        expression
    )

    expression = expression.rstrip("?.!")

    expr = sp.sympify(
        expression
    )

    result = sp.integrate(
        expr,
        x
    )

    steps = [
        f"Function: {expr}",
        f"Integral: {result} + C"
    ]

    return {
        "answer": f"{result} + C",
        "steps": steps
    }


# ============================================================
# GENERAL MATH SOLVER
# ============================================================

def solve_math(query):

    lower = query.lower().strip()

    try:

        # --------------------------------------------
        # DERIVATIVE
        # --------------------------------------------

        if (
            "derivative" in lower
            or "differentiate" in lower
        ):

            result = solve_derivative(
                query
            )

            return {
                "status": "SOLVED",
                **result
            }


        # --------------------------------------------
        # INTEGRAL
        # --------------------------------------------

        if (
            "integral" in lower
            or "integrate" in lower
        ):

            result = solve_integral(
                query
            )

            return {
                "status": "SOLVED",
                **result
            }


        # --------------------------------------------
        # EQUATION
        # --------------------------------------------

        if (
            "=" in query
            or lower.startswith("solve")
            or "find x" in lower
        ):

            result = solve_equation(
                query
            )

            if result is not None:

                return {
                    "status": "SOLVED",
                    **result
                }


        # --------------------------------------------
        # BASIC EXPRESSION
        # --------------------------------------------

        expression = clean_expression(
            query
        )

        expression = expression.rstrip(
            "?.!"
        )

        result = sp.sympify(
            expression
        )

        result = sp.simplify(
            result
        )

        return {
            "status": "SOLVED",
            "answer": str(result),
            "steps": [
                f"Expression: {expression}",
                f"Result: {result}"
            ]
        }


    except Exception as e:

        print(
            f"Math solver error: {e}"
        )

        return {
            "status": "UNABLE_TO_SOLVE",
            "answer": None,
            "steps": [],
            "error": str(e)
        }


# ============================================================
# PREDICT + SOLVE
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            force=True
        )

        query = data.get(
            "query",
            ""
        )

        if not query or not isinstance(
            query,
            str
        ):

            return jsonify({
                "error":
                'Please provide a valid math "query".'
            }), 400


        # Difficulty
        difficulty = predict_difficulty(
            query
        )


        # Actual mathematics
        solution = solve_math(
            query
        )


        return jsonify({

            "query": query,

            "predicted_difficulty":
                difficulty["difficulty"],

            "prediction_confidence":
                difficulty["confidence"],

            "class_probabilities":
                difficulty["probabilities"],

            "solution_status":
                solution["status"],

            "answer":
                solution["answer"],

            "steps":
                solution["steps"],

            "model_version":
                1.0,

            "model_accuracy":
                0.60

        }), 200


    except Exception as e:

        return jsonify({
            "error":
            f"Prediction failed: {str(e)}"
        }), 500


# ============================================================
# TRAINING LOG
# ============================================================

@app.route(
    "/api/train",
    methods=["POST"]
)
def train_append():

    try:

        data = request.get_json(
            force=True
        )

        query = data.get(
            "query",
            ""
        )

        difficulty = data.get(
            "difficulty",
            ""
        )

        if not query or not difficulty:

            return jsonify({
                "error":
                'Please provide both "query" and "difficulty".'
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
                "New pattern recorded. Ready for retraining iteration."

        }), 200


    except Exception as e:

        return jsonify({
            "error":
            f"Registration failed: {str(e)}"
        }), 500


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
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
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
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

            "train":
                "POST /api/train"

        }

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "=============================================="
    )

    print(
        "   SCALAR MATH AI API"
    )

    print(
        "=============================================="
    )

    print(
        f"Starting server on port {PORT}"
    )

    print(
        "=============================================="
    )

    app.run(
        host=HOST,
        port=PORT,
        debug=False
    )
