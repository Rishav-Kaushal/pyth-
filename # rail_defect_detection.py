# rail_defect_detection.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify

# Feature extraction function (mean and std)
def extract_features(signal):
    return [np.mean(signal), np.std(signal)]

# Generate sample dataset
def generate_data():
    np.random.seed(42)
    healthy_signals = [np.random.normal(0, 1, 100) for _ in range(50)]
    defective_signals = [np.random.normal(2, 1.5, 100) for _ in range(50)]
    healthy_features = np.array([extract_features(s) for s in healthy_signals])
    defective_features = np.array([extract_features(s) for s in defective_signals])
    X = np.vstack((healthy_features, defective_features))
    y = np.hstack((np.zeros(50), np.ones(50)))
    return X, y

# Train model and save
def train_save_model():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(clf, 'rail_defect_model.pkl')
    print("Model saved as rail_defect_model.pkl")

# Load model and predict
def predict_defect(signal):
    clf = joblib.load('rail_defect_model.pkl')
    features = extract_features(signal)
    prediction = clf.predict([features])
    return 'Defective' if prediction[0] == 1 else 'Healthy'

# Flask API for deployment
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON {"signal": [list of float values]}
    data = request.get_json()
    signal = data.get('signal', None)
    if signal is None or len(signal) == 0:
        return jsonify({'error': 'Invalid input signal'}), 400
    try:
        signal_np = np.array(signal)
        result = predict_defect(signal_np)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_save_model()
    else:
        app.run(host='0.0.0.0', port=5000)
