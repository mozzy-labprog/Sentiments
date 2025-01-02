from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load sentiment analysis pipeline (using a model with positive, negative, and neutral classes)
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="bhadresh-savani/bert-base-uncased-emotion"  # Model that supports positive, negative, and neutral
)

@app.route('/')
def home():
    # Serve the HTML file
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Perform sentiment analysis
        result = sentiment_analysis(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app in debug mode
    app.run(debug=True)
