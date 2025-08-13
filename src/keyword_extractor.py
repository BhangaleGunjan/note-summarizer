from keybert import KeyBERT
import yake
from rake_nltk import Rake
import spacy
from collections import Counter
import re
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords

class KeywordExtractor:
    """Extract keywords using multiple algorithms for comprehensive results"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize models
        self.keybert_model = None
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize RAKE
        self.rake = Rake()
        
        # Initialize YAKE
        self.yake_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Maximum number of words in keyphrase
            dedupLim=0.7,  # Deduplication threshold
            top=20  # Number of keywords to extract
        )
    
    def initialize_keybert(self):
        """Initialize KeyBERT model (lazy loading)"""
        if self.keybert_model is None:
            try:
                self.keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')
                return True
            except Exception as e:
                print(f"Could not initialize KeyBERT: {e}")
                return False
        return True
    
    def extract_keywords(self, text: str, method: str = "all", 
                        num_keywords: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords using specified method(s)
        
        Args:
            text: Input text
            method: 'keybert', 'yake', 'rake', 'spacy', or 'all'
            num_keywords: Number of keywords to return per method
            
        Returns:
            Dictionary with method names as keys and keyword lists as values
        """
        
        if len(text.strip()) < 50:
            return {"error": [("Text too short for keyword extraction", 0.0)]}
        
        results = {}
        
        # Clean text first
        cleaned_text = self._preprocess_text(text)
        
        if method == "all" or method == "keybert":
            results["keybert"] = self._extract_keybert(cleaned_text, num_keywords)
        
        if method == "all" or method == "yake":
            results["yake"] = self._extract_yake(cleaned_text, num_keywords)
        
        if method == "all" or method == "rake":
            results["rake"] = self._extract_rake(cleaned_text, num_keywords)
        
        if method == "all" or method == "spacy":
            results["spacy"] = self._extract_spacy(cleaned_text, num_keywords)
        
        # If using all methods, create a combined ranking
        if method == "all":
            results["combined"] = self._combine_results(results, num_keywords)
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and prepare text for keyword extraction"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
        
        return text.strip()
    
    def _extract_keybert(self, text: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using KeyBERT"""
        
        if not self.initialize_keybert():
            return [("KeyBERT unavailable", 0.0)]
        
        try:
            # Extract keywords with different n-gram ranges
            keywords_1 = self.keybert_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 1), 
                stop_words='english',
                top_n=num_keywords//2
            )
            
            keywords_2_3 = self.keybert_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(2, 3), 
                stop_words='english',
                top_n=num_keywords//2
            )
            
            # Combine and sort by score
            all_keywords = keywords_1 + keywords_2_3
            all_keywords.sort(key=lambda x: x[1], reverse=True)
            
            return all_keywords[:num_keywords]
            
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return [("KeyBERT failed", 0.0)]
    
    def _extract_yake(self, text: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE"""
        
        try:
            keywords = self.yake_extractor.extract_keywords(text)
            
            # YAKE returns lower scores for better keywords, so we invert
            processed_keywords = []
            for keyword, score in keywords[:num_keywords]:
                # Invert score (lower is better in YAKE)
                inverted_score = 1.0 / (1.0 + score)
                processed_keywords.append((keyword, inverted_score))
            
            return processed_keywords
            
        except Exception as e:
            print(f"YAKE extraction failed: {e}")
            return [("YAKE failed", 0.0)]
    
    def _extract_rake(self, text: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using RAKE"""
        
        try:
            self.rake.extract_keywords_from_text(text)
            keyword_scores = self.rake.get_ranked_phrases_with_scores()
            
            # RAKE returns (score, phrase) tuples
            processed_keywords = [(phrase, score) for score, phrase in keyword_scores[:num_keywords]]
            
            return processed_keywords
            
        except Exception as e:
            print(f"RAKE extraction failed: {e}")
            return [("RAKE failed", 0.0)]
    
    def _extract_spacy(self, text: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using spaCy NER and noun phrases"""
        
        try:
            doc = self.nlp(text)
            
            # Extract named entities
            entities = [(ent.text.lower(), ent.label_) for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']]
            
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                           if len(chunk.text.split()) <= 3 and len(chunk.text) > 3]
            
            # Extract important POS combinations
            important_tokens = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    important_tokens.append(token.lemma_.lower())
            
            # Count frequencies
            all_terms = [ent[0] for ent in entities] + noun_phrases + important_tokens
            term_counts = Counter(all_terms)
            
            # Convert to score format (frequency as score)
            keywords_with_scores = [(term, count) for term, count in term_counts.most_common(num_keywords)]
            
            return keywords_with_scores
            
        except Exception as e:
            print(f"spaCy extraction failed: {e}")
            return [("spaCy failed", 0.0)]
    
    def _combine_results(self, results: Dict, num_keywords: int) -> List[Tuple[str, float]]:
        """Combine results from multiple methods using weighted scoring"""
        
        combined_scores = {}
        
        # Weight different methods
        method_weights = {
            'keybert': 0.4,
            'yake': 0.3,
            'rake': 0.2,
            'spacy': 0.1
        }
        
        for method, weight in method_weights.items():
            if method in results:
                for keyword, score in results[method]:
                    if isinstance(keyword, str) and len(keyword.strip()) > 2:
                        keyword = keyword.lower().strip()
                        if keyword not in combined_scores:
                            combined_scores[keyword] = 0
                        combined_scores[keyword] += score * weight
        
        # Sort by combined score
        sorted_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords[:num_keywords]
    
    def get_keyword_relationships(self, text: str, keywords: List[str]) -> Dict[str, List[str]]:
        """Find relationships between keywords in the text"""
        
        doc = self.nlp(text)
        relationships = {keyword: [] for keyword in keywords}
        
        # Find sentences containing each keyword
        sentences = [sent.text for sent in doc.sents]
        
        for keyword in keywords:
            related_keywords = []
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    # Find other keywords in the same sentence
                    for other_keyword in keywords:
                        if (other_keyword != keyword and 
                            other_keyword.lower() in sentence.lower()):
                            related_keywords.append(other_keyword)
            
            relationships[keyword] = list(set(related_keywords))
        
        return relationships
    
    def get_summary_relationships(self, summary_text: str, concepts: List[str]) -> Dict[str, List[str]]:
        """
        Find relationships between concepts within a summary
        Summaries have more coherent structure, so we can use better methods
        """
        
        doc = self.nlp(summary_text)
        relationships = {concept: [] for concept in concepts}
        
        # Method 1: Sentence-level co-occurrence (stronger in summaries)
        sentences = [sent.text for sent in doc.sents]
        
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                relationship_strength = 0
                
                for sentence in sentences:
                    if concept1.lower() in sentence.lower() and concept2.lower() in sentence.lower():
                        relationship_strength += 3  # Strong relationship in same sentence
                        
                        # Check if they're part of the same noun phrase or clause
                        sent_doc = self.nlp(sentence)
                        concept1_tokens = [token for token in sent_doc if concept1.lower() in token.text.lower()]
                        concept2_tokens = [token for token in sent_doc if concept2.lower() in token.text.lower()]
                        
                        if concept1_tokens and concept2_tokens:
                            # Check syntactic distance
                            min_distance = min([abs(t1.i - t2.i) for t1 in concept1_tokens for t2 in concept2_tokens])
                            if min_distance <= 3:  # Very close = stronger relationship
                                relationship_strength += 2
                
                # Method 2: Sequential proximity (concepts mentioned close together)
                summary_lower = summary_text.lower()
                pos1 = summary_lower.find(concept1.lower())
                pos2 = summary_lower.find(concept2.lower())
                
                if pos1 != -1 and pos2 != -1:
                    distance = abs(pos1 - pos2)
                    if distance < 100:  # Within 100 characters
                        relationship_strength += 1
                
                # Add relationship if strong enough
                if relationship_strength >= 3:  # Threshold for summary relationships
                    relationships[concept1].append(concept2)
                    relationships[concept2].append(concept1)
        
        return relationships

    
def extract_summary_concepts(self, summary_text: str, num_concepts: int = 10) -> List[Tuple[str, float]]:
    """
    Extract concepts specifically optimized for summaries
    Focuses on entities, noun phrases, and key terms that appear in summaries
    """
    
    if len(summary_text.strip()) < 50:
        return [("Summary too short", 0.0)]
    
    doc = self.nlp(summary_text)
    concepts = []
    
    # Method 1: Named Entities (people, places, organizations, etc.)
    entities = [(ent.text.strip(), 0.9, "entity") for ent in doc.ents 
               if len(ent.text.strip()) > 2 and ent.label_ in 
               ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW']]
    
    # Method 2: Important Noun Phrases
    noun_phrases = []
    for chunk in doc.noun_chunks:
        # Filter for meaningful noun phrases
        if (len(chunk.text.split()) <= 3 and 
            len(chunk.text.strip()) > 3 and
            not chunk.text.lower().startswith(('this', 'that', 'these', 'those'))):
            noun_phrases.append((chunk.text.strip(), 0.7, "noun_phrase"))
    
    # Method 3: Key Single Terms (nouns, adjectives that appear important)
    key_terms = []
    for token in doc:
        if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
            not token.is_stop and 
            not token.is_punct and 
            len(token.text) > 3 and
            token.text.isalpha()):
            key_terms.append((token.lemma_.lower(), 0.5, "key_term"))
    
    # Method 4: Subject-Verb-Object patterns (actions/relationships)
    svo_patterns = []
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            # Look for objects
            objects = [child.text for child in token.head.children 
                      if child.dep_ in ["dobj", "pobj"]]
            
            if objects:
                for obj in objects:
                    pattern = f"{subject} {verb} {obj}"
                    if len(pattern) < 50:  # Keep it reasonable
                        svo_patterns.append((pattern, 0.8, "relationship"))
    
    # Combine all concepts
    all_concepts = entities + noun_phrases + key_terms + svo_patterns
    
    # Score and deduplicate
    concept_scores = {}
    for concept, base_score, concept_type in all_concepts:
        concept_clean = concept.lower().strip()
        
        if concept_clean not in concept_scores:
            # Boost score based on position in summary (earlier = more important)
            position_boost = 1.0
            if summary_text.lower().find(concept_clean) < len(summary_text) * 0.3:
                position_boost = 1.2
            
            # Boost score based on concept type
            type_boost = {'entity': 1.3, 'noun_phrase': 1.1, 'key_term': 1.0, 'relationship': 1.2}
            
            final_score = base_score * position_boost * type_boost.get(concept_type, 1.0)
            concept_scores[concept_clean] = (concept, final_score)
    
    # Sort by score and return top concepts
    sorted_concepts = sorted(concept_scores.values(), key=lambda x: x[1], reverse=True)
    
    return [(concept, score) for concept, score in sorted_concepts[:num_concepts]]
