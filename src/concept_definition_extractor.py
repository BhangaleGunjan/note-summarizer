import re
import spacy

class ConceptDefinitionExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_definitions(self, text):
        doc = self.nlp(text)
        definitions = {}
        
        # Pattern for "Concept is/are/refers to/means/defined as Definition"
        regex = re.compile(r"\b([A-Z][a-zA-Z0-9- ]{2,})\s+(is|are|refers to|means|can be defined as)\s+(.+?)(\.|;|$)", flags=re.IGNORECASE)
        
        for sent in doc.sents:
            match = regex.search(sent.text)
            if match:
                concept = match.group(1).strip()
                definition = match.group(3).strip()
                definitions[concept] = definition
            else:
                # Also support "Concept: definition"
                parts = sent.text.split(":")
                if len(parts) == 2 and len(parts[0].strip().split()) < 4:
                    concept = parts[0].strip()
                    definition = parts[1].strip()
                    definitions[concept] = definition
        
        return definitions
