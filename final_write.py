import os
import google.generativeai as genai
import pickle
import faiss
import numpy as np
import time
from fpdf import FPDF

def load_quotes_and_embeddings():
    """Load all quotes and generate Gemini embeddings"""
    all_quotes = []
    all_embeddings = []
    
    # Load quotes from all database files
    for file in os.listdir("databases"):
        if file.endswith("_quotes.pkl"):
            with open(os.path.join("databases", file), "rb") as f:
                quotes = pickle.load(f)
                all_quotes.extend(quotes)
                
                # Generate embeddings using correct method
                for quote in quotes:
                    # Use the proper embedding model
                    embedding = genai.embed_content(
                        model='models/embedding-001',
                        content=quote,
                        task_type="retrieval_document"
                    )['embedding']
                    all_embeddings.append(embedding)
    
    # Convert to FAISS index
    if not all_embeddings:
        raise ValueError("No quotes found in databases")
    
    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(all_embeddings, dtype=np.float32))
    
    return all_quotes, index

def get_top_quotes(query, quotes, index, top_k=3):
    """Get top matching quotes using Gemini embeddings"""
    query_embed = genai.embed_content(
        model='models/embedding-001',
        content=query,
        task_type="retrieval_query"
    )['embedding']
    
    distances, indices = index.search(np.array([query_embed], dtype=np.float32), top_k)
    return [quotes[i] for i in indices[0]]

def generate_social_isolation_essay():
    # Load analyses and quotes
    analyses_text = ""
    analysis_dir = "analyses"
    for file in os.listdir(analysis_dir):
        if file.endswith(".txt"):
            with open(os.path.join(analysis_dir, file), "r") as f:
                analyses_text += f"\n\n### {file} Analysis:\n{f.read()}"
    
    quotes, index = load_quotes_and_embeddings()
    
    # Define theme queries for each paragraph
    theme_queries = {
        'introduction': "Universal nature of social isolation in modern literature",
        'body1': "Social isolation manifested through character relationships",
        'body2': "Societal structures enforcing isolation",
        'body3': "Psychological impacts of long-term isolation", 
        'conclusion': "Synthesis of isolation's role in human condition"
    }
    
    # Get quotes for each theme
    theme_quotes = {section: get_top_quotes(query, quotes, index) 
                   for section, query in theme_queries.items()}
    
    # Build the essay prompt
    prompt = f"""Write a sophisticated 5-paragraph essay analyzing social isolation. Structure:

1. Introduction ({theme_queries['introduction']}):
   - Use quotes: {', '.join([f'"{q}"' for q in theme_quotes['introduction'][:1]])}

2. Body 1 - Relationships ({theme_queries['body1']}):
   - Analyze: {', '.join([f'"{q}"' for q in theme_quotes['body1'][:3]])}

3. Body 2 - Society ({theme_queries['body2']}):
   - Examine: {', '.join([f'"{q}"' for q in theme_quotes['body2'][:3]])}

4. Body 3 - Psychology ({theme_queries['body3']}):
   - Discuss: {', '.join([f'"{q}"' for q in theme_quotes['body3'][:3]])}

5. Conclusion ({theme_queries['conclusion']}):
   - Synthesize using: {', '.join([f'"{q}"' for q in theme_quotes['conclusion'][:1]])}

Analyses Context:
{analyses_text}

Maintain academic tone while directly analyzing quotes. Highlight connections between different narratives."""
    
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_API_KEY") or input("Enter Google AI API key: ")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text

def txt_to_pdf(txt_file, pdf_file):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    # Open the text file
    with open(txt_file, "r") as file:
        for line in file:
            pdf.cell(200, 10, txt=line, ln=True)

    # Output the pdf
    pdf.output(pdf_file)

def save_essay(essay_text):
    # Save essay with enhanced formatting
    with open("social_isolation_analysis.txt", "w") as f:
        f.write(essay_text)
        
    txt_to_pdf("social_isolation_analysis.txt", "social_isolation_analysis.pdf")


if __name__ == "__main__":
    essay = generate_social_isolation_essay()
    save_essay(essay)