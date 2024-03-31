#!/usr/bin/env python3
import sys

# Function to calculate relevance scores between query and documents
def calculate_relevance(query_tfidf, doc_tfidf):
    relevance_score = sum(query_weight * doc_tfidf.get(idx, 0) for idx, query_weight in query_tfidf)
    return relevance_score

# Initialize variables
query_tfidf = None
doc_tfidf = {}

# Read input and calculate relevance scores
for line in sys.stdin:
    line = line.strip()
    if line.startswith("query"):
        _, query_vector = line.split('\t', 1)
        query_tfidf = eval(query_vector)  # Convert string representation of list back to list
    else:
        doc_id, doc_vector = line.split('\t', 1)
        doc_tfidf[doc_id] = eval(doc_vector)  # Convert string representation of list back to list

# Calculate relevance scores and emit final results
print("Relevance scores:")
for doc_id, doc_vector in doc_tfidf.items():
    relevance_score = calculate_relevance(query_tfidf, doc_vector)
    print(f"doc {doc_id}: {relevance_score}")

