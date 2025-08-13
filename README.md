# 🧠 NeuroNotes

> **AI-powered interactive learning from your notes summarize, visualize, master.**  
> Works **fully offline**, free, and open-source.  
> Extracts text from your notes (PDF, text, or images), generates summaries, keywords, and builds interactive concept maps including **definition-based maps** for real understanding.

---

## Features

- **Upload** PDF, TXT, PNG/JPG, BMP, TIFF.
- **Extract text** (OCR for images using Tesseract).
- **Summarize** (multiple free, local AI models: transformer, extractive, hybrid).
- **Keyword extraction** (KeyBERT, YAKE, RAKE, spaCy).
- **Definition extraction**
- **Interactive concept maps** (keyword-based or definition-based).
- Works **completely offline** 	no paid APIs.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BhangaleGunjan/note-summarizer.git
cd neuronotes
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv neuronotes_env
.
neuronotes_env\Scripts\activate

# macOS/Linux
python3 -m venv neuronotes_env
source neuronotes_env/bin/activate
```

### 3. Install Core Requirements
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Download Required Models/Data
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> **Note:** For OCR, install Tesseract:  
> - Windows: [Tesseract Download](https://github.com/tesseract-ocr/tesseract/wiki)  
> - macOS: `brew install tesseract`  
> - Linux: `sudo apt install tesseract-ocr`

---

## Project Structure

```
NeuroNotes/
├── main.py                # Streamlit main app entry point
├── requirements.txt       # Python requirements
├── README.md
├── docs/                  # Extended documentation (see below)
├── src/
│   ├── __init__.py
│   ├── file_handler.py
│   ├── text_extractor.py
│   ├── summarizer.py
│   ├── keyword_extractor.py
│   ├── concept_definition_extractor.py
│   ├── concept_mapper.py
├── data/
│   ├── uploads/           # Temporary files
│   └── exports/           # Generated concept maps, reports
├── static/
│   ├── neuronotes_logo.png
│   ├── illustration_dragdrop.png
│   └── template.html      # (optional, for pyvis custom templates)
```

---

## Usage Guide

#### 0. Activate Environment
```bash
# Windows
neuronotes_env\Scripts\activate
# macOS/Linux
source neuronotes_env/bin/activate
```

#### 1. Start the App
```bash
streamlit run main.py
```

#### 2. Upload a Document
- PDF, TXT, PNG/JPG/BMP/TIFF images, or Markdown.
- Click **Analyze** after upload.

#### 3. Explore Results
- **Summary Tab:** AI-generated notes, original text expander, method/length metrics.
- **Keywords Tab:** Table/List view; optionally compare extraction algorithms.
- **Concept Map Tab:** Drag/zoom; download map as HTML. If using Definition Map, hover nodes for definition popups.
- **Analytics Tab:** Stats word count, central nodes, keyword relationships.
- **Export Tab:** Download JSON/CSV reports, saved maps.

#### 4. Settings & Customization

In sidebar:
- Choose summarization and keyword extraction methods.
- Optionally toggle between Keyword and Definition - concept map modes.
- Advanced: Show process steps; enable quiz generator (if implemented).

---

## Key Components

- **File Handler:** Handles file uploads, saving, cleaning up.
- **Text Extractor:** Uses PyMuPDF/pdfplumber/pytesseract/Pillow/OpenCV for robust text extraction.
- **Summarizer:** Selects between transformer models, extractive summarizers, hybrid approaches.
- **Keyword Extractor:** Fuses KeyBERT, YAKE, RAKE, spaCy for semantic/statistical extraction.
- **Concept Definition Extractor:** Scans text for definition patterns (e.g. Linux is an operating system. Python: programming language).
- **Concept Mapper:** Visualizes concepts as nodes; definitions as tooltips/labels; connections via inter-concept references.

---

## Advanced Features

- **Definition-based Mapping:** See definitions as hover-text or beside nodes.
- **Quiz Generator:** Automatically creates MCQs from content/concepts. (Under development)
- **PDF/CSV/JSON Export:** Download everything for offline use or sharing.
- **Custom Templates:** Improve concept map visuals by customizing Pyvis HTML template.
- **Fast Error Handling:** Clear UI feedback for common import/OCR/model errors.

---

## License

Licensed under the **MIT License** free to use, modify, and distribute.

---

## Credits

Developed by Gunjan Bhangale.
Core libraries: Streamlit, PyMuPDF, pdfplumber, pytesseract, spaCy, sumy, transformers, KeyBERT, yake, rake-nltk, NetworkX, Pyvis, Pillow, OpenCV.

---
