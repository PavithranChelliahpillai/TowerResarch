# save_as: analysis_generator.py
import os
import cohere
import time
import pickle
import faiss
import numpy as np
from typing import List, Tuple
import traceback
from sentence_transformers import SentenceTransformer

co = cohere.Client(os.environ['COHERE_API_KEY'])
model = SentenceTransformer('all-MiniLM-L6-v2')  # This produces 384-dimensional vectors

def load_quotes_and_index(base_name: str) -> Tuple[List[str], faiss.IndexFlatL2]:
    quotes_path = os.path.join("databases", f"{base_name}_quotes.pkl")
    index_path = os.path.join("databases", f"{base_name}_index.faiss")
    
    try:
        # Load quotes from pickle file
        with open(quotes_path, "rb") as f:
            quotes = pickle.load(f)
            
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        return quotes, index
    except Exception as e:
        print(f"Error loading data for {base_name}: {str(e)}")
        return [], None

def get_top_quotes(base_name: str, summary: str, top_k: int = 3) -> List[str]:
    # Retrieve top-k most relevant quotes using semantic similarity
    # Args:
    #     base_name: Book name without extensions
    #     summary: Book summary text
    #     top_k: Number of quotes to return
    # Returns:
    #     List of top relevant quotes
    quotes, index = load_quotes_and_index(base_name)
    if not quotes or index is None:
        print(f"Could not load quotes or index for {base_name}")
        return []
    
    # Get the dimension of the FAISS index
    index_dimension = index.d
    print(f"FAISS index dimension: {index_dimension}")
    
    # Generate summary embedding using the SAME model used in preprocessing
    try:
        # Use sentence_transformers instead of Cohere for embedding
        summary_embed = model.encode([summary], convert_to_numpy=True)
        summary_embed = summary_embed.astype(np.float32)  # Convert to float32 for FAISS
        
        # Print dimensions for debugging
        print(f"Embedding shape: {summary_embed.shape}")
        
        # Check dimensions
        if summary_embed.shape[1] != index_dimension:
            print(f"ERROR: Embedding dimension ({summary_embed.shape[1]}) doesn't match index dimension ({index_dimension})")
            return []
        
        # Search FAISS index
        distances, indices = index.search(summary_embed, min(top_k, len(quotes)))
        
        # Return quotes with index bounds checking
        result_quotes = []
        for i in indices[0]:
            if 0 <= i < len(quotes):
                result_quotes.append(quotes[i])
            else:
                print(f"Warning: Index {i} out of bounds for quotes list of length {len(quotes)}")
        
        print(f"Found {len(result_quotes)} quotes for {base_name}")
        return result_quotes
        
    except Exception as e:
        print(f"Error in get_top_quotes: {e}")
        traceback.print_exc()  # More detailed error information
        return []

def generate_thematic_analysis(summary: str) -> str:
    # Generate analysis focused on social isolation theme
    # Args:
    #     summary: Book summary text
    # Returns:
    #     Thematic analysis text
    if not summary:
        print("Empty summary provided to thematic analysis")
        return ""
        
    prompt = f"""Analyze how the story explores social isolation through:
1. Character relationships and alienation
2. Physical/emotional barriers in the narrative
3. Societal structures impacting characters
4. Symbolic representations of isolation

Consider both explicit and implicit manifestations of isolation.

Story Summary:
{summary}

In-depth Analysis:"""
    
    try:
        response = co.chat(
            message=prompt,
            model="command-r-plus-08-2024",
            temperature=0.4,
            max_tokens=600  # Add token limit to prevent excessively long responses
        )
        result = response.text.strip()
        if not result:
            print("Received empty response from thematic analysis")
        return result
    except Exception as e:
        print(f"Thematic analysis failed: {e}")
        traceback.print_exc()
        return ""

def generate_integrated_analysis(summary: str, quotes: List[str]) -> str:
    # Generate comprehensive analysis combining summary and quotes
    # Args:
    #     summary: Book summary text
    #     quotes: List of relevant quotes
    # Returns:
    #     Integrated analysis text
    if not summary:
        print("Empty summary provided to integrated analysis")
        return ""
        
    if not quotes:
        print("No quotes provided to integrated analysis")
        return ""
    
    # Join quotes with better formatting
    formatted_quotes = "\n".join([f'• "{q}"' for q in quotes])
    
    prompt = f"""Synthesize a sophisticated literary analysis that:
1. Begins with the theme of social isolation
2. Analyzes {len(quotes)} key quotes in context
3. Explores philosophical implications
4. Maintains academic tone

Structure:
- Introduction: Establish thematic context
- Body: Integrated quote analysis
- Conclusion: Broader significance

Quotes:
{formatted_quotes}

Story Context:
{summary}

Comprehensive Analysis:"""
    
    try:
        response = co.generate(
            model="command-r-plus-08-2024",
            prompt=prompt,
            max_tokens=650,
            temperature=0.5,
            stop_sequences=["\n\n\n"]
        )
        result = response.generations[0].text.strip()
        if not result:
            print("Received empty response from integrated analysis")
        return result
    except Exception as e:
        print(f"Integrated analysis failed: {e}")
        traceback.print_exc()
        return ""

def process_book(summary_path: str):
    # Process a single book's analysis pipeline
    # Args:
    #     summary_path: Path to summary file
    # Derive base name from filename
    base_name = os.path.basename(summary_path).replace("_summary.txt", "")
    
    # Read summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read().strip()
            
        if not summary:
            print(f"Warning: {summary_path} contains empty summary")
            return
            
    except Exception as e:
        print(f"Error reading {summary_path}: {e}")
        return
    
    # Generate analyses
    try:
        print(f"Processing {base_name}...")
        
        # Thematic analysis
        print(f"Generating thematic analysis for {base_name}...")
        thematic = generate_thematic_analysis(summary)
        if not thematic:
            print(f"Warning: Empty thematic analysis for {base_name}")
        
        # Quote selection and integrated analysis
        print(f"Retrieving quotes for {base_name}...")
        quotes = get_top_quotes(base_name, summary)
        
        if not quotes:
            print(f"Warning: No quotes found for {base_name}")
            quotes = []  # Ensure it's an empty list, not None
        
        print(f"Found {len(quotes)} quotes for analysis")
            
        if thematic and quotes:
            print(f"Generating integrated analysis for {base_name}...")
            integrated = generate_integrated_analysis(summary, quotes)
        else:
            print(f"Skipping integrated analysis due to missing thematic analysis or quotes")
            integrated = ""
        
        # Create output directory if it doesn't exist
        output_dir = "analyses"
        os.makedirs(output_dir, exist_ok=True)
        
        # Write outputs if they exist
        # if thematic:
        #     # Write thematic analysis
        #     with open(os.path.join(output_dir, f"{base_name}_thematic.txt"), "w", encoding="utf-8") as f:
        #         f.write(thematic)
        #     print(f"Saved thematic analysis for {base_name}")
        
        # Write combined analysis if both components exist
        if thematic or integrated:
            with open(os.path.join(output_dir, f"{base_name}_full_analysis.txt"), "w", encoding="utf-8") as f:
                if thematic:
                    f.write(f"Thematic Analysis:\n{thematic}\n\n")
                if integrated:
                    f.write(f"Integrated Analysis:\n{integrated}")
            print(f"Saved full analysis for {base_name}")
        
        print(f"Completed processing {base_name}")
            
    except Exception as e:
        print(f"Failed to process {base_name}: {e}")
        traceback.print_exc()

def main():
    # Check for Cohere API key
    if not os.environ.get("COHERE_API_KEY"):
        print("ERROR: COHERE_API_KEY environment variable not set")
        print("Please set it with: export COHERE_API_KEY='your-api-key'")
        return
    
    # Validate directory structure
    required_dirs = ["summaries", "databases"]
    missing_dirs = []
    
    for d in required_dirs:
        if not os.path.exists(d):
            missing_dirs.append(d)
    
    if missing_dirs:
        print(f"ERROR: Missing required directories: {', '.join(missing_dirs)}")
        print("Please create these directories before running the script")
        return
    
    # Process all summary files
    summary_files = [f for f in os.listdir("summaries") if f.endswith("_summary.txt")]
    
    if not summary_files:
        print("No summary files found in 'summaries' directory")
        print("Please add summary files in the format 'bookname_summary.txt'")
        return
    
    print(f"Found {len(summary_files)} summary files to process")
    
    for fname in summary_files:
        process_book(os.path.join("summaries", fname))
    
    print("Analysis generation completed!")

if __name__ == "__main__":
    main()