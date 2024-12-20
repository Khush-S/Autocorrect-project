import pandas as pd
import re
import textdistance
from collections import Counter

class TextProcessor:
    def __init__(self, file_path):
        self.words = self.read_and_process_text(file_path)
        self.V = set(self.words)
        self.word_freq_dict = Counter(self.words)
        self.bigrams = list(zip(self.words[:-1], self.words[1:]))
        self.bigram_freq_dict = Counter(self.bigrams)
        self.probs = {k: v / sum(self.word_freq_dict.values()) for k, v in self.word_freq_dict.items()}
        self.bigram_probs = {k: v / self.word_freq_dict[k[0]] for k, v in self.bigram_freq_dict.items()}

    def read_and_process_text(self, file_path):
        """Reads and processes the text file with UTF-8 encoding."""
        words = []
        with open(file_path, 'r', encoding='utf-8') as f:
            file_name_data = f.read().lower()  # Convert to lowercase for case-insensitive processing
            words = re.findall(r'\w+', file_name_data)  # Extract words using regex
        return words

    def corrections(self, input_word):
        """Function to autocorrect a single word."""
        input_word = input_word.lower()
        if input_word in self.V:
            return input_word  # If the word is correct, return it as is.
        else:
            # Compute similarity scores for corrections using Jaccard distance
            similarities = [1 - textdistance.Jaccard(qval=2).distance(v, input_word) for v in self.word_freq_dict.keys()]
            
            # Create a DataFrame with probabilities and similarity scores
            df = pd.DataFrame.from_dict(self.probs, orient='index').reset_index()
            df = df.rename(columns={'index': 'Word', 0: 'Prob'})
            df['Similarity'] = similarities

            # Sort by similarity and probability
            top_match = df.sort_values(['Similarity', 'Prob'], ascending=False).iloc[0]
            return top_match['Word']

    def corrections_and_predictions(self, input_sentence, suggestion_count=5):
        """Function to autocorrect a sentence and provide next word suggestions."""
        input_sentence = input_sentence.lower()
        input_words = re.findall(r'\w+', input_sentence)
        corrected_sentence = []

        # Correct the input words
        for word in input_words:
            corrected_sentence.append(self.corrections(word))

        # Predict the next word with multiple suggestions
        last_word = corrected_sentence[-1] if corrected_sentence else None
        suggestions = []
        if last_word:
            # Filter bigrams that start with the last word and sort by probability
            suggestions = {k[1]: v for k, v in self.bigram_probs.items() if k[0] == last_word}
            suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)[:suggestion_count]

        return ' '.join(corrected_sentence), suggestions
