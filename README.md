# 🧠 NeuroNotes

> **AI-powered interactive learning from your notes 	6 summarize, visualize, master.**  
> Works **fully offline**, free, and open-source.  
> Extracts text from your notes (PDF, text, or images), generates summaries, keywords, and builds interactive concept maps 	6 including **definition-based maps** for real understanding.

---

## F680 Features

- 4C2 **Upload** PDF, TXT, PNG/JPG, BMP, TIFF.
- 50D **Extract text** (OCR for images using Tesseract).
- 4DD **Summarize** (multiple free, local AI models: transformer, extractive, hybrid).
- 511 **Keyword extraction** (KeyBERT, YAKE, RAKE, spaCy).
- 4DA **Definition extraction** (	3Concept: Definition	4 for educational maps).
- 570 **Interactive concept maps** (keyword-based or definition-based).
- 3AF **Quiz generation** (optional).
- 4BE Works **completely offline** 	6 no paid APIs.

---

## 4E6 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/neuronotes.git
cd neuronotes
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv neuronotes_env
.
euronotes_env\Scriptsctivate

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

## 4C2 Project Structure

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
│   ├── quiz_generator.py  # (optional)
│   └── pdf_exporter.py    # (optional)
├── data/
│   ├── uploads/           # Temporary files
│   └── exports/           # Generated concept maps, reports
├── static/
│   ├── neuronotes_logo.png
│   ├── illustration_dragdrop.png
│   └── template.html      # (optional, for pyvis custom templates)
```

---

## 449 Usage Guide

#### 0. Activate Environment
```bash
# Windows
.
euronotes_env\Scriptsctivate
# macOS/Linux
source neuronotes_env/bin/activate
```

#### 1. Start the App
```bash
streamlit run main.py
```

#### 2. Upload a Document
- PDF, TXT, PNG/JPG/BMP/TIFF images, or Markdown.
- Click **680 Analyze** after upload.

#### 3. Explore Results
- **Summary Tab:** AI-generated notes, original text expander, method/length metrics.
- **Keywords Tab:** Table/List view; optionally compare extraction algorithms.
- **Concept Map Tab:** Drag/zoom; download map as HTML. If using 	3Definition Map,	4 hover nodes for definition popups.
- **Analytics Tab:** Stats	6word count, central nodes, keyword relationships.
- **Export Tab:** Download JSON/CSV reports, saved maps.

#### 4. Settings & Customization

In sidebar:
- Choose summarization and keyword extraction methods.
- Optionally toggle between 	3Keyword	4 and 	3Definition	4 concept map modes.
- Advanced: Show process steps; enable quiz generator (if implemented).

---

## 4D6 Key Components

- **File Handler:** Handles file uploads, saving, cleaning up.
- **Text Extractor:** Uses PyMuPDF/pdfplumber/pytesseract/Pillow/OpenCV for robust text extraction.
- **Summarizer:** Selects between transformer models, extractive summarizers, hybrid approaches.
- **Keyword Extractor:** Fuses KeyBERT, YAKE, RAKE, spaCy for semantic/statistical extraction.
- **Concept Definition Extractor:** Scans text for definition patterns (e.g. 	1inux is an operating system.	2, 	1ython: programming language	2).
- **Concept Mapper:** Visualizes concepts as nodes; definitions as tooltips/labels; connections via inter-concept references.

---

## 4E6 Advanced Features

- **Definition-based Mapping:** See definitions as hover-text or beside nodes.
- **Quiz Generator:** Automatically creates MCQs from content/concepts.
- **PDF/CSV/JSON Export:** Download everything for offline use or sharing.
- **Custom Templates:** Improve concept map visuals by customizing Pyvis HTML template.
- **Fast Error Handling:** Clear UI feedback for common import/OCR/model errors.

---

## 6E0 Troubleshooting

| Problem                        | Solution |
|--------------------------------|----------|
| **Blank UI:** See README for `__init__.py` and import path fixes; ensure all dependencies in requirements.txt are installed. |
| **Module Import Errors:** See installation instructions above for Tesseract/unusual packages. |
| **PyMuPDF/cv2 Import:** Confirm installed in active venv: `pip install pymupdf opencv-python` |
| **spaCy model Missing:** Run `python -m spacy download en_core_web_sm` |
| **Tesseract not Found:** Install Tesseract binary and update path in `text_extractor.py` if needed. |
| **Concept Map Misbehavior:** Lower number of keywords, try definition-based map, check for updates. |

---

## 4C4 License

Licensed under the **MIT License** 	6 free to use, modify, and distribute.

---

## 64F Credits

Developed by [Your Name] with guidance from Perplexity AI.
Core libraries: Streamlit, PyMuPDF, pdfplumber, pytesseract, spaCy, sumy, transformers, KeyBERT, yake, rake-nltk, NetworkX, Pyvis, Pillow, OpenCV.

---

## 4D1 docs/EXTENDED_DOC.md (Optional)

For your extended docs folder, create an EXTENDED_DOC.md with:
- Technical specs & pipeline details
- Developer guide
- IPR information
- Future feature ideas

