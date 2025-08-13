import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, BinaryIO

class FileHandler:
    """Handles file uploads and basic file operations"""
    
    def __init__(self):
        self.supported_extensions = {
            'pdf': ['.pdf'],
            'image': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
            'text': ['.txt', '.md', '.rtf']
        }
    
    def upload_file(self) -> Optional[Tuple[str, str, BinaryIO]]:
        """
        Display file uploader and return file info
        Returns: (filename, file_type, file_object) or None
        """
        uploaded_file = st.file_uploader(
            "Choose a file to analyze",
            type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'md', 'bmp', 'tiff'],
            help="Upload PDF documents, images with text, or text files"
        )
        
        if uploaded_file is not None:
            file_type = self._determine_file_type(uploaded_file.name)
            return uploaded_file.name, file_type, uploaded_file
        
        return None
    
    def _determine_file_type(self, filename: str) -> str:
        """Determine file type based on extension"""
        file_ext = Path(filename).suffix.lower()
        
        for file_type, extensions in self.supported_extensions.items():
            if file_ext in extensions:
                return file_type
        
        return 'unknown'
    
    def save_temp_file(self, uploaded_file: BinaryIO, filename: str) -> str:
        """Save uploaded file temporarily and return path"""
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / filename
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return str(temp_path)
    
    def cleanup_temp_file(self, file_path: str):
        """Remove temporary file"""
        try:
            os.remove(file_path)
        except OSError:
            pass  # File might already be deleted
