from flask import Flask, request, jsonify, Flask
import joblib
import threading
import time
import requests

app = Flask(__name__)
# 1. Load the original joblib file
base_pipeline = joblib.load('/content/math_difficulty_model.joblib')

# 2. Package it with metadata (scalar parameter metrics)
scalar_model_package = {
    'model': base_pipeline,
    'version': 1.0,
    'accuracy_scalar': 0.60,
    'num_features_scalar': 1000
}

# Save the scalar-packaged joblib
joblib.dump(scalar_model_package, '/content/scalar_math_difficulty_model.joblib')

# 3. Create Flask API on port 5004
app_5004 = Flask("ScalarModelAPI")

@app_5004.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Please provide a valid math "query" in your request body.'}), 400
        
        # Load package dynamically and predict
        pkg = joblib.load('/content/scalar_math_difficulty_model.joblib')
        prediction = pkg['model'].predict([query])[0]
        
        return jsonify({
            'query': query,
            'predicted_difficulty': prediction.upper(),
            'model_version': pkg['version'],
            'model_accuracy': pkg['accuracy_scalar']
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app_5004.route('/api/train', methods=['POST'])
def train_append():
    """
    Endpoint allowing users to submit new query-difficulty mappings to simulate live training
    """
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        difficulty = data.get('difficulty', '')
        
        if not query or not difficulty:
            return jsonify({'error': 'Please provide both "query" and "difficulty" to register training.'}), 400
            
        # Simulate online update log
        with open('/content/training_logs.txt', 'a') as log_file:
            log_file.write(f"{query}|||{difficulty}\n")
            
        return jsonify({
            'status': 'success',
            'message': 'New pattern recorded. Ready for retraining iteration.'
        })
    except Exception as e:
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

def run_api_5004():
    app_5004.run(host='0.0.0.0', port=5004, debug=False, use_reloader=False)

# Launch background server thread
flask_thread_5004 = threading.Thread(target=run_api_5004)
flask_thread_5004.daemon = True
flask_thread_5004.start()

time.sleep(1.5)
print("Scalar Model API successfully running on http://127.0.0.1:5004")

if __name__ == "__main__":
    app.run(debug=True)
