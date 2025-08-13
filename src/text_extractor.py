try:
    import fitz  # Most common import
except ModuleNotFoundError:
    # PyMuPDF 1.23+ also offers 'pymupdf' alias
    import pymupdf as fitz
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging

class TextExtractor:
    """Extracts text from various file formats using multiple methods"""
    
    def __init__(self):
        # Configure tesseract path if needed (Windows users)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """Main method to extract text based on file type"""
        
        if file_type == 'pdf':
            return self._extract_from_pdf(file_path)
        elif file_type == 'image':
            return self._extract_from_image(file_path)
        elif file_type == 'text':
            return self._extract_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF with fallback to pdfplumber"""
        
        # Method 1: Try PyMuPDF first (faster)
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            
            doc.close()
            
            if text.strip():  # If we got text, return it
                return text.strip()
                
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed: {e}")
        
        # Method 2: Fallback to pdfplumber for complex layouts
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                return text.strip()
                
        except Exception as e:
            self.logger.error(f"Both PDF extraction methods failed: {e}")
            return ""
    
    def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with preprocessing"""
        
        try:
            # Load and preprocess image for better OCR
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to improve OCR accuracy
            # Remove noise
            denoised = cv2.medianBlur(gray, 5)
            
            # Increase contrast
            contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = contrast.apply(denoised)
            
            # Convert back to PIL Image for tesseract
            pil_image = Image.fromarray(enhanced)
            
            # Extract text using tesseract with configuration
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;: '
            text = pytesseract.image_to_string(pil_image, config=custom_config)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _extract_from_text(self, text_path: str) -> str:
        """Extract text from plain text files"""
        
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
                
        except UnicodeDecodeError:
            # Try different encoding
            try:
                with open(text_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                self.logger.error(f"Text file reading failed: {e}")
                return ""
