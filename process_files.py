# Pre-process all files into TXT format, with the same name as the original file.
# Supported formats: PDF, EPUB, XML

import os
import fitz  
from ebooklib import epub
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

def extract_epub(file_path):
    book = epub.read_epub(file_path)
    text = []
    for item in book.get_items():
        if item.get_type() == 9:  
            soup = BeautifulSoup(item.content, "html.parser")
            text.append(soup.get_text())
    return "\n".join(text)

def extract_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    text = []
    for elem in root.iter():
        if elem.text:
            text.append(elem.text.strip())
    
    return "\n".join(text)

def convert(directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if filename.endswith(".txt"):
            output_path = os.path.join(output_directory, filename)
            os.rename(file_path, output_path)  # Move TXT files as-is
            continue

        base_name = os.path.splitext(filename)[0] 
        output_path = os.path.join(output_directory, f"{base_name}.txt")

        try:
            if filename.endswith(".pdf"):
                text = extract_pdf(file_path)
            elif filename.endswith(".epub"):
                text = extract_epub(file_path)
            elif filename.endswith(".xml"):
                text = extract_xml(file_path)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Converted {filename} → {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

convert("books/", "converted_books/")
