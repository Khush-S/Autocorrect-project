from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import textdistance
from collections import Counter

app = Flask(__name__)

# Reading and processing the text file with UTF-8 encoding
words = []
with open('book.txt', 'r', encoding='utf-8') as f:  # Specify encoding as 'utf-8'
    file_name_data = f.read()
    file_name_data = file_name_data.lower()
    words = re.findall(r'\w+', file_name_data)  # Use raw string for regex

# Vocabulary and frequency dictionary
V = set(words)
word_freq_dict = Counter(words)

# Creating bigram frequency dictionary
bigrams = list(zip(words[:-1], words[1:]))
bigram_freq_dict = Counter(bigrams)

# Calculate probabilities for unigrams and bigrams
probs = {k: v / sum(word_freq_dict.values()) for k, v in word_freq_dict.items()}
bigram_probs = {k: v / word_freq_dict[k[0]] for k, v in bigram_freq_dict.items()}

def my_autocorrect(input_word):
    """Function to autocorrect a single word."""
    input_word = input_word.lower()
    if input_word in V:
        return input_word  # If the word is correct, return it as is.
    else:
        # Compute similarity scores for corrections
        similarities = [1 - textdistance.Jaccard(qval=2).distance(v, input_word) for v in word_freq_dict.keys()]
        
        # Create a DataFrame with probabilities and similarity scores
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities

        # Sort by similarity and probability
        top_match = df.sort_values(['Similarity', 'Prob'], ascending=False).iloc[0]
        return top_match['Word']

def my_autocorrect_and_autocomplete(input_sentence, suggestion_count=5):
    """Function to autocorrect a sentence and provide next word suggestions."""
    input_sentence = input_sentence.lower()
    input_words = re.findall(r'\w+', input_sentence)  # Use raw string for regex
    corrected_sentence = []

    # Correct the input words
    for word in input_words:
        corrected_sentence.append(my_autocorrect(word))

    # Predict the next word with multiple suggestions
    last_word = corrected_sentence[-1] if corrected_sentence else None
    suggestions = []
    if last_word:
        # Filter bigrams that start with the last word and sort by probability
        suggestions = {k[1]: v for k, v in bigram_probs.items() if k[0] == last_word}
        suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:suggestion_count]

    # Return the corrected sentence and the list of suggested words
    return ' '.join(corrected_sentence), suggestions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    input_sentence = request.form['input_sentence']
    
    corrected_sentence, suggestions = my_autocorrect_and_autocomplete(input_sentence)

    # Prepare response
    suggestion_words = [{'word': word, 'prob': prob} for word, prob in suggestions]

    return jsonify({
        'corrected_sentence': corrected_sentence,
        'suggestions': suggestion_words
    })

if __name__ == '__main__':
    app.run(debug=True)
