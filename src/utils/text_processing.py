import re
import string
from typing import List, Dict, Any, Set, Optional
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextProcessor:
    """Handles text processing and feature extraction"""
    
    def __init__(self):
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common e-commerce and real estate terms
        self.domain_stopwords = {
            'listing', 'item', 'product', 'property', 'home', 'house',
            'available', 'sale', 'rent', 'buy', 'purchase', 'offer'
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    pass
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """Extract important keywords from text"""
        if not text:
            return []
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords and punctuation
        keywords = [
            token for token in tokens 
            if token not in self.stop_words 
            and token not in self.domain_stopwords
            and token not in string.punctuation
            and len(token) > 2
            and token.isalpha()
        ]
        
        # Lemmatize
        keywords = [self.lemmatizer.lemmatize(word) for word in keywords]
        
        # Get word frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:max_keywords]]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if not text:
            return {}
        
        entities = {
            'prices': [],
            'addresses': [],
            'phone_numbers': [],
            'dates': [],
            'dimensions': [],
            'brands': [],
            'numbers': []
        }
        
        # Extract prices
        price_patterns = [
            r'\$[\d,]+\.?\d*[kK]?',
            r'USD?\s*[\d,]+\.?\d*',
            r'[\d,]+\.?\d*\s*dollars?'
        ]
        for pattern in price_patterns:
            entities['prices'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        entities['phone_numbers'].extend(re.findall(phone_pattern, text))
        
        # Extract dimensions/measurements
        dimension_patterns = [
            r'\d+\.?\d*\s*x\s*\d+\.?\d*\s*(?:ft|feet|sq|square)?',
            r'\d+\.?\d*\s*(?:sq\.?\s*)?(?:ft|feet|meters?|m\b)',
            r'\d+\.?\d*\s*(?:bed|bath|car)',
        ]
        for pattern in dimension_patterns:
            entities['dimensions'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract years/dates
        year_pattern = r'\b(19|20)\d{2}\b'
        entities['dates'].extend(re.findall(year_pattern, text))
        
        # Extract general numbers
        number_pattern = r'\b\d+\.?\d*\b'
        entities['numbers'].extend(re.findall(number_pattern, text))
        
        return entities
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores for text"""
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1 (negative to positive)
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1 (objective to subjective)
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from text"""
        if not text:
            return {}
        
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'keywords': self.extract_keywords(text),
            'entities': self.extract_entities(text),
            'sentiment': self.calculate_sentiment(text),
            'has_caps': any(c.isupper() for c in text),
            'has_numbers': any(c.isdigit() for c in text),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?')
        }
        
        return features
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(self.extract_keywords(text1))
        words2 = set(self.extract_keywords(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Extract text matching specific regex patterns"""
        if not text or not patterns:
            return []
        
        matches = []
        for pattern in patterns:
            try:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            except re.error:
                continue
        
        return matches
    
    def contains_keywords(self, text: str, keywords: List[str], min_matches: int = 1) -> bool:
        """Check if text contains specified keywords"""
        if not text or not keywords:
            return False
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        return matches >= min_matches