import spacy
import PyPDF2
import re
from typing import Dict, List, Any
import streamlit as st
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import concurrent.futures
import torch

class OptimizedLegalAnalyzer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load models efficiently
        self.load_models()
        
        # Precompile regex patterns for better performance
        self.compile_patterns()

    def load_models(self):
        """Load models efficiently"""
        try:
            # Use smaller SpaCy model optimized for speed
            self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'vectors'])
            # Add sentencizer for sentence boundary detection
            self.nlp.add_pipe('sentencizer')
            
            # Load efficient sentence transformer model
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create embeddings for common legal terms and sections
            self.section_embeddings = self._create_section_embeddings()
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    def compile_patterns(self):
        """Precompile regex patterns for better performance"""
        self.patterns = {
            'dates': re.compile(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})|(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})', re.IGNORECASE),
            'money': re.compile(r'Rs\.?\s*([\d,]+(?:\.\d{2})?)|(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:Rupees?|INR)', re.IGNORECASE),
            'names': re.compile(r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            'addresses': re.compile(r'(?:residing|located)\s+at[:\s]+([^,\n]+(?:,\s*[^,\n]+){2,})', re.IGNORECASE),
            'pan': re.compile(r'PAN\s*:?\s*([A-Z]{5}[0-9]{4}[A-Z])', re.IGNORECASE),
            'sections': re.compile(r'(?:^|\n\n)\s*([A-Z][A-Z\s]+)(?:\s*:|\.)(?:\s|\n)+', re.MULTILINE)
        }

    def _create_section_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for common legal sections"""
        common_sections = [
            "parties and definitions",
            "term and termination",
            "payment terms",
            "obligations and responsibilities",
            "property details",
            "governing law",
            "confidentiality",
            "indemnification"
        ]
        
        embeddings = self.sentence_model.encode(common_sections)
        return dict(zip(common_sections, embeddings))

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF efficiently"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Use parallel processing for large documents
            text_parts = []
            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())
            
            text = " ".join(text_parts)
            
            # Efficient text cleaning
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
        except Exception as e:
            self.logger.error(f"PDF extraction error: {str(e)}")
            return None

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Main analysis function"""
        try:
            # Break text into sections
            sections = self._split_into_sections(text)
            
            # Process sections
            results = defaultdict(dict)
            for section_name, section_text in sections.items():
                section_info = self._process_section(section_text)
                results[section_name] = section_info
            
            # Post-process and organize results
            final_results = self._organize_results(dict(results))
            return final_results
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return None

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into logical sections using semantic similarity"""
        sections = {}
        current_section = []
        current_section_name = "general"
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # Get paragraph embedding
            para_embedding = self.sentence_model.encode(para[:512])
            
            # Find most similar section
            max_similarity = 0
            best_section = current_section_name
            
            for section_name, section_embedding in self.section_embeddings.items():
                similarity = np.dot(para_embedding, section_embedding) / (
                    np.linalg.norm(para_embedding) * np.linalg.norm(section_embedding)
                )
                if similarity > max_similarity and similarity > 0.5:
                    max_similarity = similarity
                    best_section = section_name
            
            if best_section != current_section_name:
                if current_section:
                    sections[current_section_name] = ' '.join(current_section)
                current_section = []
                current_section_name = best_section
            
            current_section.append(para)
        
        if current_section:
            sections[current_section_name] = ' '.join(current_section)
            
        return sections

    def _process_section(self, text: str) -> Dict[str, Any]:
        """Process individual sections efficiently"""
        results = {}
        
        # Extract structured information using regex
        for key, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            results[key] = [m.group() for m in matches]
        
        # Extract key phrases and important sentences
        doc = self.nlp(text)
        important_sentences = []
        
        for sent in doc.sents:
            # Check for important legal keywords
            if any(keyword in sent.text.lower() for keyword in ['shall', 'must', 'agree', 'terminate', 'payment']):
                important_sentences.append(sent.text)
        
        if important_sentences:
            results['key_points'] = important_sentences
        
        return results

    def _organize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize and clean results"""
        organized = defaultdict(list)
        
        # Organize by type
        for section, content in results.items():
            if not content:
                continue
                
            for key, items in content.items():
                if not items:
                    continue
                    
                if key == 'dates':
                    organized['important_dates'].extend(items)
                elif key == 'money':
                    organized['financial_terms'].extend(items)
                elif key == 'names':
                    organized['parties'].extend(items)
                elif key == 'addresses':
                    organized['locations'].extend(items)
                elif key == 'key_points':
                    organized['key_terms'].extend(items)
                else:
                    organized[key].extend(items)
        
        # Remove duplicates while preserving order
        final_results = {}
        for key, items in organized.items():
            if isinstance(items, list):
                final_results[key] = list(dict.fromkeys(items))
            else:
                final_results[key] = items
        
        return final_results

def main():
    st.title("Smart Legal Document Analyzer")
    st.write("Upload any legal document (PDF) to extract key information")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        analyzer = OptimizedLegalAnalyzer()
        
        with st.spinner("Analyzing document..."):
            # Extract text
            text = analyzer.extract_text_from_pdf(uploaded_file)
            
            if text:
                # Analyze document
                info = analyzer.analyze_document(text)
                
                if info:
                    st.success("Analysis complete!")
                    
                    # Display results
                    for section, content in info.items():
                        with st.expander(f"{section.replace('_', ' ').title()}"):
                            if isinstance(content, list):
                                for item in content:
                                    st.write(f"• {item}")
                            else:
                                st.write(content)

if __name__ == "__main__":
    main()