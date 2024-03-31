#!/usr/bin/env python3
import sys
from collections import Counter

# Function to find word frequencies in a given text
def frequency_finder(text):
    words = text.split()
    word_frequencies = Counter(words)
    return word_frequencies

# Function to calculate TF/IDF weights for a given text
def calculate_tfidf(text, vocabulary, IDF_format):
    tfidf_data = []
    word_frequencies = frequency_finder(text)
    for word, freq in word_frequencies.items():
        if word in vocabulary:
            term_index = vocabulary[word]
            tfidf = freq * IDF_format[term_index]
            tfidf_data.append((term_index, tfidf))
    return tfidf_data

if __name__ == "__main__":
    # Read vocabulary and IDF weights from input
    vocabulary = {}
    IDF_format = []
    for line in sys.stdin:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue  # Skip lines that don't contain expected key-value pairs
        key, value = parts
        if key == "vocabulary":
            vocabulary = eval(value)
        elif key == "IDF_format":
            IDF_format = eval(value)
        elif key == "query":
            # Calculate TF/IDF weights for the query
            query_text = value
            query_tfidf = calculate_tfidf(query_text, vocabulary, IDF_format)
            # Emit query vector separately
            print(f"query\t{query_tfidf}")
        else:
            # Process document
            doc_id = key
            text = value
            # Calculate TF/IDF weights for the document
            tfidf_vector = calculate_tfidf(text, vocabulary, IDF_format)
            # Emit intermediate key-value pairs from mapper
            print(f"{doc_id}\t{tfidf_vector}")
