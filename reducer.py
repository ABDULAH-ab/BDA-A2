#!/usr/bin/env python
import sys
from collections import Counter
import math

# Function to calculate TF
def frequency_finder(t):
    words = t.split()
    word_f = Counter(words)
    return word_f

# Read data from mapper and create a dictionary with ID and section text
data = {}
for line in sys.stdin:
    line = line.strip()
    ID, text = line.split(',', 1)
    data[ID] = text

# Create vocabulary
vocabulary = set()
for text in data.values():
    vocabulary.update(text.split())

# Store vocabulary in the form of index and word as key
vocab = {word: index for index, word in enumerate(vocabulary)}

# Calculate TF
TF = {}
for doc_id, text in data.items():
    word_freq = frequency_finder(text)
    tf = [(vocab[word], freq) for word, freq in word_freq.items() if word in vocab]
    TF[doc_id] = tf

# Calculate IDF
# Counting the number of documents (or sentences) in your corpus
total_documents = len(data)

# Counting the number of documents that contain each word in the vocabulary
document_frequency = {word: 0 for word in vocabulary}
for word in vocabulary:
    for text in data.values():
        if word in text:
            document_frequency[word] += 1

# Calculating IDF for each word in the vocabulary
IDF = {}
for word, doc_freq in document_frequency.items():
    idf_score = math.log10(total_documents / (doc_freq + 1))  # Adding 1 to avoid division by zero
    IDF[word] = idf_score

# Sorting the vocabulary based on the index
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

# Storing IDF scores in the desired format (as a dictionary)
IDF_format = {idx: IDF[word] for word, idx in sorted_vocab}


# TF/IDF Weights
# Calculating TF/IDF weights and store them in the specified format
TF_IDF_weights = {}

# Iterating over each document
for doc_id, tf_data in TF.items():
    tfidf_data = []
    
    # Calculating TF/IDF for each term in the document
    for term_index, tf in tf_data:
        tfidf = tf * IDF_format[term_index]  # TF/IDF calculation
        tfidf_data.append((term_index, tfidf))
    
    # Sorting by term index
    tfidf_data.sort(key=lambda x: x[0])
    
    # Storing TF/IDF weights in the specified format
    tfidf_format = ", ".join(f"({idx}, {weight:.2f})" for idx, weight in tfidf_data)
    TF_IDF_weights[doc_id] = tfidf_format

# Print TF/IDF weights for each document
#for doc_id, tfidf_format in TF_IDF_weights.items():
    #print(doc_id, tfidf_format)
    #print(tfidf_format)

# Vector Space Model
# Creating a sparse vector representation for each document
vector_space = {}

# Iterating over each document
for doc_id, tf_data in TF.items():
    sparse_vector = []
    
    # Iterating over TF/IDF data for the document
    for term_index, tf in tf_data:
        tfidf = tf * IDF_format[term_index]  # Calculating TF/IDF
        if tfidf != 0:
            sparse_vector.append((term_index, tfidf))  # Adding non-zero elements
        
    # Storing sparse vector representation
    vector_space[doc_id] = sparse_vector

# Print sparse vector representation for each document
#for doc_id, sparse_vector in vector_space.items():
#    vector_str = ", ".join(f"({idx}: {weight})" for idx, weight in sparse_vector)
#    print(f"The sparse vector representation for document {doc_id} is as follows:")
#    print("[", vector_str, "]")

