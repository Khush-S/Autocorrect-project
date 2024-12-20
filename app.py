from flask import Flask, render_template, request, jsonify
from utils import TextProcessor  # Import the TextProcessor class

app = Flask(__name__)

# Initialize the TextProcessor class with the text file
text_processor = TextProcessor('book.txt')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    input_sentence = request.form['input_sentence']
    
    corrected_sentence, suggestions = text_processor.corrections_and_predictions(input_sentence)

    # Prepare response
    suggestion_words = [{'word': word, 'prob': prob} for word, prob in suggestions]

    return jsonify({
        'corrected_sentence': corrected_sentence,
        'suggestions': suggestion_words
    })

if __name__ == '__main__':
    app.run(debug=True)
