import unittest
import tempfile
import os
from pathlib import Path

# Import your modules
from src.text_extractor import TextExtractor
from src.summarizer import TextSummarizer
from src.keyword_extractor import KeywordExtractor

class TestNeuroNotesBasics(unittest.TestCase):
    """Basic functionality tests for NeuroNotes components"""
    
    def setUp(self):
        self.sample_text = """
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn and make decisions from data without being explicitly 
        programmed. It involves algorithms that can identify patterns in data 
        and make predictions or classifications based on those patterns.
        """
        
        self.text_extractor = TextExtractor()
        self.summarizer = TextSummarizer()
        self.keyword_extractor = KeywordExtractor()
    
    def test_text_extraction_from_text_file(self):
        """Test text extraction from a simple text file"""
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.sample_text)
            temp_path = f.name
        
        try:
            extracted_text = self.text_extractor.extract_text(temp_path, 'text')
            self.assertIn("machine learning", extracted_text.lower())
            self.assertGreater(len(extracted_text), 50)
        finally:
            os.unlink(temp_path)
    
    def test_summarization(self):
        """Test text summarization functionality"""
        
        result = self.summarizer.summarize_text(self.sample_text, method="extractive")
        
        self.assertIsInstance(result, dict)
        self.assertIn('summary', result)
        self.assertIn('method', result)
        self.assertGreater(len(result['summary']), 0)
    
    def test_keyword_extraction(self):
        """Test keyword extraction functionality"""
        
        results = self.keyword_extractor.extract_keywords(self.sample_text, method="rake", num_keywords=5)
        
        self.assertIsInstance(results, dict)
        self.assertIn('rake', results)
        
        keywords = results['rake']
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        
        # Check keyword format
        for keyword, score in keywords:
            self.assertIsInstance(keyword, str)
            self.assertIsInstance(score, (int, float))

if __name__ == '__main__':
    unittest.main()
