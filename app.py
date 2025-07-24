import streamlit as st
import yake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os
import random
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

st.set_page_config(page_title="NeuroNotes 🧠", layout="wide")

st.title("🧠 NeuroNotes")
st.write("Welcome to your AI-powered study assistant. Let's turn messy notes into clean brain fuel 💥")

uploaded_file = st.file_uploader("📤 Upload your notes (PDF, TXT, or Image)", type=["pdf", "txt", "png", "jpg", "jpeg"])

def summarize_text(text, sentence_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join([str(sentence) for sentence in summary])

def extract_keywords(text, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(n=1, top=max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [word for word, score in keywords]

def generate_concept_map(keywords):
    G = nx.Graph()

    # Add central node (NeuroNotes) like a sexy sun
    G.add_node("NeuroNotes")

    # Add each keyword as connected to the center
    for keyword in keywords:
        G.add_node(keyword)
        G.add_edge("NeuroNotes", keyword)

    # Create Pyvis network
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.3)

    # Save and render in temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)
    
    return tmp_file.name

def generate_mcqs(text, keywords, num_questions=3):
    questions = []

    # Split text into sentences
    sentences = sent_tokenize(text)

    # Filter sentences that contain keywords
    candidate_sentences = [s for s in sentences if any(kw.lower() in s.lower() for kw in keywords)]

    for _ in range(num_questions):
        if not candidate_sentences:
            break

        sent = random.choice(candidate_sentences)
        candidate_sentences.remove(sent)

        used_kw = None
        for kw in keywords:
            if kw.lower() in sent.lower():
                used_kw = kw
                break

        if not used_kw:
            continue

        # Replace keyword with blank
        question = re.sub(rf"\b{re.escape(used_kw)}\b", "_____", sent, flags=re.IGNORECASE)

        # Generate 3 fake options
        distractors = random.sample([kw for kw in keywords if kw != used_kw], k=min(3, len(keywords)-1))
        options = distractors + [used_kw]
        random.shuffle(options)

        questions.append({
            "question": question,
            "options": options,
            "answer": used_kw
        })

    return questions

if uploaded_file:
    st.success("File uploaded successfully!")
    file_details = {
        "Filename": uploaded_file.name,
        "FileType": uploaded_file.type,
        "Size (KB)": round(uploaded_file.size / 1024, 2)
    }
    st.json(file_details)
    if st.button("🔍 Process File"):
        file_type = uploaded_file.type

        extracted_text = ""

        # Process TXT files
        if file_type == "text/plain":
            extracted_text = uploaded_file.read().decode("utf-8")

        # Process PDFs
        elif file_type == "application/pdf":
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() or ""

        # Process Images
        elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
            from PIL import Image
            import pytesseract
            image = Image.open(uploaded_file)
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            extracted_text = pytesseract.image_to_string(image)

        else:
            st.error("Unsupported file type.")

        if extracted_text:
            st.subheader("📝 Extracted Text")
            st.text_area("Here's what I found:", extracted_text, height=300)
            st.subheader("📌 Summary (Top 5 Points)")
            summary = summarize_text(extracted_text)
            st.write(summary)
            st.subheader("🔑 Keywords")
            keywords = extract_keywords(extracted_text)
            st.write(", ".join(keywords))
            st.subheader("🕸️ Concept Map")
            map_file = generate_concept_map(keywords)
            with open(map_file, 'r', encoding='utf-8') as f:
                map_html = f.read()
            components.html(map_html, height=550, scrolling=True)

        else:
            st.warning("No text could be extracted. Try another file?")
