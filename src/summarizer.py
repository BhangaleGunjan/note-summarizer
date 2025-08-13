from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import spacy
import torch
from typing import List, Dict
import re

class TextSummarizer:
    """Multi-method text summarization using free AI models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nlp = None
        self.transformer_pipeline = None
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Please install spaCy English model: python -m spacy download en_core_web_sm")
    
    def initialize_transformer(self, model_name: str = "facebook/bart-large-cnn"):
        """Initialize HuggingFace transformer model"""
        
        try:
            # Use a lightweight model for better performance
            model_name = "sshleifer/distilbart-cnn-12-6"  # Smaller, faster model
            
            self.transformer_pipeline = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            
            return True
            
        except Exception as e:
            print(f"Could not load transformer model: {e}")
            return False
    
    def summarize_text(self, text: str, method: str = "hybrid", 
                      summary_length: str = "medium") -> Dict[str, str]:
        """
        Summarize text using specified method
        
        Args:
            text: Input text to summarize
            method: 'transformer', 'extractive', or 'hybrid'
            summary_length: 'short', 'medium', or 'long'
            
        Returns:
            Dictionary with summary and metadata
        """
        
        if len(text.strip()) < 100:
            return {
                "summary": text,
                "method": "original",
                "word_count": len(text.split()),
                "compression_ratio": 1.0
            }
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        if method == "transformer":
            return self._transformer_summarize(cleaned_text, summary_length)
        elif method == "extractive":
            return self._extractive_summarize(cleaned_text, summary_length)
        else:  # hybrid
            return self._hybrid_summarize(cleaned_text, summary_length)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and prepare text for summarization"""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove very short sentences (likely noise)
        sentences = text.split('.')
        filtered_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return '. '.join(filtered_sentences)
    
    def _transformer_summarize(self, text: str, length: str) -> Dict[str, str]:
        """Summarize using HuggingFace transformer"""
        
        if self.transformer_pipeline is None:
            if not self.initialize_transformer():
                # Fallback to extractive method
                return self._extractive_summarize(text, length)
        
        try:
            # Set parameters based on desired length
            max_length, min_length = self._get_length_params(text, length)
            
            # Handle long texts by chunking
            if len(text.split()) > 1000:
                chunks = self._chunk_text(text, 800)
                summaries = []
                
                for chunk in chunks:
                    result = self.transformer_pipeline(
                        chunk,
                        max_length=max_length//len(chunks),
                        min_length=min_length//len(chunks),
                        do_sample=False
                    )
                    summaries.append(result[0]['summary_text'])
                
                final_summary = ' '.join(summaries)
            else:
                result = self.transformer_pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                final_summary = result[0]['summary_text']
            
            return {
                "summary": final_summary,
                "method": "transformer",
                "word_count": len(final_summary.split()),
                "compression_ratio": len(text.split()) / len(final_summary.split())
            }
            
        except Exception as e:
            print(f"Transformer summarization failed: {e}")
            return self._extractive_summarize(text, length)
    
    def _extractive_summarize(self, text: str, length: str) -> Dict[str, str]:
        """Extractive summarization using SUMY"""
        
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            
            # Use LSA summarizer (works well for academic content)
            summarizer = LsaSummarizer()
            summarizer.stop_words = self._get_stop_words()
            
            sentence_count = self._get_sentence_count(text, length)
            summary_sentences = summarizer(parser.document, sentence_count)
            
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            
            return {
                "summary": summary,
                "method": "extractive",
                "word_count": len(summary.split()),
                "compression_ratio": len(text.split()) / len(summary.split())
            }
            
        except Exception as e:
            print(f"Extractive summarization failed: {e}")
            # Return first few sentences as fallback
            sentences = text.split('.')[:5]
            fallback_summary = '. '.join(sentences) + '.'
            
            return {
                "summary": fallback_summary,
                "method": "fallback",
                "word_count": len(fallback_summary.split()),
                "compression_ratio": len(text.split()) / len(fallback_summary.split())
            }
    
    def _hybrid_summarize(self, text: str, length: str) -> Dict[str, str]:
        """Combine extractive and transformer methods"""
        
        # First, use extractive to get key sentences
        extractive_result = self._extractive_summarize(text, "long")
        intermediate_text = extractive_result["summary"]
        
        # Then use transformer to refine
        if len(intermediate_text.split()) > 50:
            transformer_result = self._transformer_summarize(intermediate_text, length)
            transformer_result["method"] = "hybrid"
            return transformer_result
        else:
            extractive_result["method"] = "hybrid"
            return extractive_result
    
    def _chunk_text(self, text: str, max_words: int) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            chunks.append(chunk)
        
        return chunks
    
    def _get_length_params(self, text: str, length: str):
        """Get max/min length parameters based on text and desired length"""
        
        text_length = len(text.split())
        
        if length == "short":
            max_len = min(100, text_length // 4)
            min_len = min(50, max_len // 2)
        elif length == "medium":
            max_len = min(200, text_length // 3)
            min_len = min(100, max_len // 2)
        else:  # long
            max_len = min(400, text_length // 2)
            min_len = min(200, max_len // 2)
        
        return max_len, min_len
    
    def _get_sentence_count(self, text: str, length: str) -> int:
        """Determine number of sentences for extractive summary"""
        
        total_sentences = len(text.split('.'))
        
        if length == "short":
            return min(3, max(1, total_sentences // 10))
        elif length == "medium":
            return min(5, max(2, total_sentences // 8))
        else:  # long
            return min(8, max(3, total_sentences // 5))
    
    def _get_stop_words(self):
        """Get stop words for summarization"""
        try:
            from sumy.nlp.stemmers import Stemmer
            from sumy.utils import get_stop_words
            return get_stop_words("english")
        except:
            # Fallback stop words
            return ["the", "is", "at", "which", "on", "a", "an", "and", "or", "but", "in", "with", "to", "for", "of", "as", "by"]
