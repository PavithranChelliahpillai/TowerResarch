import re
import cohere
import time
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Initialize models and clients
co = cohere.Client(os.environ['COHERE_API_KEY'])
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create output directories
os.makedirs("databases", exist_ok=True)
os.makedirs("summaries", exist_ok=True)

def merge_small_paragraphs(input_content, min_words=1000):
    # Merge paragraphs into chunks of at least min_words, prioritizing newline splits, then sentence splits.
    # Read input content
    if isinstance(input_content, str) and os.path.exists(input_content):
        with open(input_content, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = input_content

    # Split and merge paragraphs first
    paragraphs = re.split(r"\n+", text)
    merged_paragraphs = []
    buffer = []
    current_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_words = len(para.split())
        
        if current_words + para_words >= min_words:
            # Commit the buffer with this paragraph
            if buffer:
                merged_paragraphs.append(" ".join(buffer + [para]).strip())
                buffer = []
                current_words = 0
            else:
                # Handle single large paragraph
                merged_paragraphs.append(para)
            continue
            
        buffer.append(para)
        current_words += para_words

    # Handle remaining content
    if buffer:
        merged_paragraphs.append(" ".join(buffer).strip())

    # Split overlarge paragraphs using sentence boundaries
    final_paragraphs = []
    split_threshold = min_words * 1.5  # Only split chunks significantly larger than min_words
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z]|$)"

    for chunk in merged_paragraphs:
        chunk_words = len(chunk.split())
        
        if chunk_words < split_threshold:
            final_paragraphs.append(chunk)
            continue
            
        # Split into sentences and re-merge
        sentences = re.split(sentence_pattern, chunk)
        sentence_buffer = []
        current_words = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words >= min_words:
                if sentence_buffer:
                    final_paragraphs.append(" ".join(sentence_buffer).strip())
                    sentence_buffer = []
                    current_words = 0
                sentence_buffer.append(sentence)
                current_words += sentence_words
            else:
                sentence_buffer.append(sentence)
                current_words += sentence_words

        if sentence_buffer:
            final_paragraphs.append(" ".join(sentence_buffer).strip())

    return final_paragraphs

def summarize_text(text):
    # Summarize text using Cohere's API
    response = co.summarize(
        model='command',
        text=text,
    )
    return response.summary

def cosine_similarity(a, b):
    # Calculate cosine similarity between two vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_key_quotes(paragraphs, top_n=3, lambda_param=0.7):
    # Extract key quotes using MMR algorithm
    key_quotes = []
    for para in paragraphs:
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", para)
        if len(sentences) <= top_n:
            key_quotes.extend(sentences)
            continue

        sentence_embeddings = model.encode(sentences)
        para_embedding = np.mean(sentence_embeddings, axis=0)
        
        similarities_to_para = [cosine_similarity(embed, para_embedding) for embed in sentence_embeddings]
        
        selected_indices = []
        candidates = list(range(len(sentences)))

        for _ in range(top_n):
            mmr_scores = []
            for cand_idx in candidates:
                relevance = similarities_to_para[cand_idx]
                redundancy = 0
                
                if selected_indices:
                    redundancy = max(
                        cosine_similarity(sentence_embeddings[cand_idx], sentence_embeddings[sel_idx])
                        for sel_idx in selected_indices
                    )

                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                mmr_scores.append(mmr_score)

            best_idx = candidates[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            candidates.remove(best_idx)

        selected_sentences = [sentences[i] for i in sorted(selected_indices)]
        key_quotes.extend(selected_sentences)
    
    return key_quotes

def store_vectors_in_faiss(vectors, sentences):
    # Store vectors in FAISS index
    vectors_np = np.array(vectors).astype('float32')
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)
    return index, sentences

def process_paragraphs_and_store(input_file, base_name):
    # Process and store book-specific databases
    ret_array = merge_small_paragraphs(input_file)
    
    all_sentences = []
    all_vectors = []

    for para in ret_array:
        key_quotes = extract_key_quotes([para])
        encoded_quotes = model.encode(key_quotes)
        
        all_vectors.extend(encoded_quotes)
        all_sentences.extend(key_quotes)

    faiss_index, faiss_sentences = store_vectors_in_faiss(all_vectors, all_sentences)

    with open(f"databases/{base_name}_quotes.pkl", "wb") as f:
        pickle.dump(faiss_sentences, f)

    faiss.write_index(faiss_index, f"databases/{base_name}_index.faiss")
    print(f"Saved {len(faiss_sentences)} quotes for {base_name}")

    return faiss_index, faiss_sentences

def generate_summary(input_file, base_name):
    # Generate and save book summary
    ret_array = merge_small_paragraphs(input_file)
    
    iteration = 0
    while len(ret_array) > 1:
        iteration += 1
        print(f"Summary iteration {iteration} for {base_name}")
        new_ret_array = []
        
        for i, para in enumerate(ret_array):
            summary = summarize_text(para)
            print(summary)
            new_ret_array.append(summary)
        
        ret_array = merge_small_paragraphs("\n".join(new_ret_array))

    summary_text = ret_array[0]
    with open(f"summaries/{base_name}_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    
    return summary_text

def process_book(input_file_path):
    # Process a single book file
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    print(f"\nProcessing {base_name}...")
    book_db, book_quotes = process_paragraphs_and_store(input_file_path, base_name)
    
    print(f"Summarizing {base_name}...")
    final_summary = generate_summary(input_file_path, base_name)
    
    return book_db, book_quotes, final_summary

if __name__ == "__main__":
    converted_books_dir = "converted_books"
    
    for book_file in os.listdir(converted_books_dir):
        if book_file.endswith(".txt"):
            book_path = os.path.join(converted_books_dir, book_file)
            db, quotes, summary = process_book(book_path)
            
            print(f"\nCompleted processing for {book_file}")
            print(f"Summary saved to: summaries/{os.path.splitext(book_file)[0]}_summary.txt")
            print(f"Database files saved in databases/ directory with prefix {os.path.splitext(book_file)[0]}")