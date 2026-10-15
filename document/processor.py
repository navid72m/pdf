"""
Document processing utilities for text extraction and chunking.
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Class to handle document processing, chunking, and text extraction.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_mode: str = "auto",
        min_chunk_size: int = 200,  # Minimum chunk size to enforce
        max_single_chunk_size: int = 10000  # Maximum size for a single chunk
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            chunking_mode: Chunking strategy ("auto", "character", "paragraph", "sentence", "whole")
            min_chunk_size: Minimum size of a chunk in characters
            max_single_chunk_size: Maximum size when using whole document as a single chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_mode = chunking_mode
        self.min_chunk_size = min_chunk_size
        self.max_single_chunk_size = max_single_chunk_size
        self.is_summary_task = False  # Flag for summary tasks
    
    def set_summary_mode(self, enabled: bool = True):
        """
        Set whether this is a summary task.
        For summary tasks, we optimize for keeping more content in fewer chunks.
        
        Args:
            enabled: Whether to enable summary optimization
        """
        self.is_summary_task = enabled
        logger.info(f"Summary mode {'enabled' if enabled else 'disabled'}")
    
    def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a document file: extract text and chunk it.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata about the document
            task_type: Optional task type ('summary', 'qa', etc.)
            
        Returns:
            Tuple of (list of text chunks, list of metadata dictionaries)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Check if this is a summary task
        if task_type == "summary":
            self.set_summary_mode(True)
        
        # Extract text from document
        logger.info(f"Processing document: {file_path}")
        text = self._extract_text(file_path)
        
        if not text:
            logger.warning(f"No text could be extracted from {file_path}")
            return [], []
        
        # Clean up the text - remove excessive whitespace and normalize
        text = self._preprocess_text(text)
        logger.info(f"Extracted and preprocessed text, total length: {len(text)} characters")
        
        # For summary tasks with small documents, use the whole document
        if self.is_summary_task and len(text) <= self.max_single_chunk_size:
            logger.info(f"Summary task with small document ({len(text)} chars), using whole document")
            chunks = [text]
        # For regular small documents, also consider using the whole document
        elif len(text) <= self.max_single_chunk_size // 2:
            logger.info(f"Small document ({len(text)} chars), using whole document")
            chunks = [text]
        else:
            # Create base metadata if not provided
            base_metadata = {"source": os.path.basename(file_path)}
            if metadata:
                base_metadata.update(metadata)
                
            # Try to enhance metadata with extracted information
            try:
                from .metadata_extractor import MetadataExtractor
                extractor = MetadataExtractor()
                base_metadata = extractor.extract_metadata(
                    file_path=file_path,
                    text=text,
                    existing_metadata=base_metadata
                )
            except (ImportError, Exception) as e:
                logger.debug(f"Metadata extraction not available: {e}")
            
            # Choose chunking strategy based on task and document size
            effective_chunk_size = self.chunk_size
            effective_chunking_mode = self.chunking_mode
            
            # For summary tasks, use larger chunks with less overlap
            if self.is_summary_task:
                effective_chunk_size = min(5000, self.chunk_size * 2)
                effective_chunking_mode = "paragraph"
                logger.info(f"Using optimized chunking for summary task: size={effective_chunk_size}, mode={effective_chunking_mode}")
            
            # Chunk the document based on selected strategy
            if effective_chunking_mode == "paragraph":
                logger.info(f"Using paragraph-based chunking with min_chunk_size={self.min_chunk_size}")
                chunks = self._chunk_by_paragraphs(text, min_chunk_size=self.min_chunk_size)
            elif effective_chunking_mode == "sentence":
                logger.info(f"Using sentence-based chunking with chunk_size={effective_chunk_size}, overlap={self.chunk_overlap}")
                chunks = self._chunk_by_sentences(text, effective_chunk_size, self.chunk_overlap)
            elif effective_chunking_mode == "character":
                logger.info(f"Using character-based chunking with chunk_size={effective_chunk_size}, overlap={self.chunk_overlap}")
                chunks = self._chunk_text(text, effective_chunk_size, self.chunk_overlap)
            elif effective_chunking_mode == "whole":
                logger.info("Using whole document as a single chunk")
                # Split into sections if document is too large
                if len(text) > self.max_single_chunk_size:
                    chunks = self._split_large_document(text)
                else:
                    chunks = [text]
            else:  # "auto" mode - choose best strategy based on text
                if len(text) < 10000:  # For shorter texts, use paragraph chunking
                    logger.info(f"Auto-selected paragraph chunking (text length: {len(text)})")
                    chunks = self._chunk_by_paragraphs(text, min_chunk_size=self.min_chunk_size)
                else:
                    logger.info(f"Auto-selected character chunking (text length: {len(text)})")
                    chunks = self._chunk_text(text, effective_chunk_size, self.chunk_overlap)
        
        # Filter out any chunks that are too small (could be fragments)
        original_chunk_count = len(chunks)
        chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_size]
        
        if len(chunks) < original_chunk_count:
            logger.info(f"Filtered out {original_chunk_count - len(chunks)} chunks that were too small (< {self.min_chunk_size} chars)")
        
        # If we have no chunks after filtering, use a different approach
        if not chunks and text:
            logger.warning("No chunks passed size filter, falling back to forced chunking")
            chunks = self._force_chunk_text(text, self.chunk_size)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Log stats about chunks for debugging
        if chunks:
            chunk_sizes = [len(chunk) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunks)
            logger.info(f"Average chunk size: {avg_size:.1f} characters")
            logger.info(f"Smallest chunk: {min(chunk_sizes)} characters, Largest chunk: {max(chunk_sizes)} characters")
        
        # Create base metadata if not provided
        if 'base_metadata' not in locals():
            base_metadata = {"source": os.path.basename(file_path)}
            if metadata:
                base_metadata.update(metadata)
        
        # Create chunk-specific metadata
        chunk_metadata = []
        document_id = base_metadata.get("document_id") or str(uuid.uuid4())
        for i, chunk in enumerate(chunks):
            # Create unique ID for document chunk
            doc_id = document_id
            
            # Create metadata with document info and chunk position
            metadata_item = {
                **base_metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "document_id": doc_id,
                "chunk_position": f"{i+1}/{len(chunks)}",
                "chunk_size_chars": len(chunk),
                "chunk_size_words": len(chunk.split())
            }
            chunk_metadata.append(metadata_item)
        
        # Reset summary mode
        self.set_summary_mode(False)
        
        return chunks, chunk_metadata
    
    def _split_large_document(self, text: str) -> List[str]:
        """
        Split a large document into manageable pieces without strict chunking.
        Used for summary tasks with large documents.
        
        Args:
            text: Document text
            
        Returns:
            List of document sections
        """
        # Find major section breaks
        section_breaks = []
        
        # Look for common section headers
        section_patterns = [
            r'\n\s*#{1,3}\s+([^\n]+)',  # Markdown headers
            r'\n\s*([A-Z][A-Z\s]{2,})\s*\n',  # ALL CAPS HEADINGS
            r'\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}:)',  # Title Case Headings with colon
            r'\n\s*(\d+\.?\s+[A-Z][a-z]+)',  # Numbered sections
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                section_breaks.append(match.start())
        
        # If no section breaks found, fall back to even splits
        if not section_breaks and len(text) > self.max_single_chunk_size:
            # Calculate how many chunks we need
            num_chunks = (len(text) + self.max_single_chunk_size - 1) // self.max_single_chunk_size
            # Create even splits
            chunk_size = len(text) // num_chunks
            # Create break points
            for i in range(1, num_chunks):
                pos = i * chunk_size
                # Try to find a sentence break nearby
                for offset in range(200):
                    if pos + offset < len(text) and text[pos + offset] in ['.', '!', '?']:
                        section_breaks.append(pos + offset + 1)
                        break
                    if pos - offset > 0 and text[pos - offset] in ['.', '!', '?']:
                        section_breaks.append(pos - offset + 1)
                        break
                else:
                    # No sentence break found, use exact position
                    section_breaks.append(pos)
        
        # Sort breaks and split text
        section_breaks.sort()
        
        if not section_breaks:
            return [text]
            
        chunks = []
        start_pos = 0
        
        for break_pos in section_breaks:
            if break_pos - start_pos >= self.min_chunk_size:
                chunks.append(text[start_pos:break_pos].strip())
                start_pos = break_pos
        
        # Add final chunk
        if len(text) - start_pos >= self.min_chunk_size:
            chunks.append(text[start_pos:].strip())
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Fix anomalies that could interfere with chunking
        text = text.replace('\r\n', '\n')  # Normalize line endings
        
        # Replace repeated newlines with a single newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove strange control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Fix common PDF extraction artifacts
        text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)  # Fix hyphenation
        
        return text.strip()
    
    def _extract_text(self, file_path: str) -> str:
        """
        Extract text from a document file based on its extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif ext == '.txt':
            return self._extract_text_from_txt(file_path)
        elif ext == '.md':
            return self._extract_text_from_txt(file_path)
        elif ext == '.docx':
            return self._extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        # Try PyMuPDF first if available (better extraction)
        try:
            import fitz  # PyMuPDF
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF using PyMuPDF")
            return text
        except ImportError:
            logger.info("PyMuPDF not installed, falling back to PyPDF2")
        except Exception as e:
            logger.warning(f"Error using PyMuPDF: {e}, falling back to PyPDF2")
        
        # Fall back to PyPDF2
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is not installed. "
                "Please install it with `pip install PyPDF2`."
            )
        
        text = ""
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                num_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
        
        logger.info(f"Extracted {len(text)} characters from PDF using PyPDF2")
        return text
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Extracted text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Extracted {len(text)} characters from text file")
            return text
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return ""
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text
        """
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx is not installed. "
                "Please install it with `pip install python-docx`."
            )
        
        try:
            doc = docx.Document(file_path)
            text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
            
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {e}")
            return ""
    
    def _chunk_by_paragraphs(self, text: str, max_chunks: int = 15, min_chunk_size: int = 200) -> List[str]:
        """
        Split text into chunks by paragraphs, ensuring reasonable chunk sizes.
        
        Args:
            text: The text to chunk
            max_chunks: Target maximum number of chunks to create
            min_chunk_size: Minimum size for a chunk in characters
            
        Returns:
            List of text chunks
        """
        logger.info(f"Paragraph chunking with max_chunks={max_chunks}, min_chunk_size={min_chunk_size}")
        
        # Find paragraph boundaries using multiple possible separators
        split_points = []
        
        # Look for double line breaks (traditional paragraphs)
        for match in re.finditer(r'\n\s*\n', text):
            split_points.append(match.start())
        
        # Look for sentences ending with period followed by space and capital letter
        for match in re.finditer(r'\.(?=\s+[A-Z])', text):
            split_points.append(match.start() + 1)  # Include the period
        
        # Sort split points by position
        split_points.sort()
        
        # Now split the text at these points
        if not split_points:
            if len(text) >= min_chunk_size:
                return [text]
            else:
                return []
                
        paragraphs = []
        pending = []
        last_pos = 0
        
        for pos in split_points:
            paragraph = text[last_pos:pos].strip()
            if pending:
                paragraph = " ".join(pending + [paragraph]).strip()
                pending = []
            if paragraph and len(paragraph) >= min_chunk_size:
                paragraphs.append(paragraph)
            elif paragraph and paragraphs and len(paragraphs[-1]) + len(paragraph) <= self.chunk_size:
                # Combine with previous paragraph if it's too small
                paragraphs[-1] = paragraphs[-1] + " " + paragraph
            elif paragraph:
                # Keep track of small paragraphs that we might want to combine later
                pending.append(paragraph)
                
            last_pos = pos + 1
        
        # Don't forget the last paragraph
        last_paragraph = " ".join(pending + [text[last_pos:].strip()]).strip()
        if last_paragraph and len(last_paragraph) >= min_chunk_size:
            paragraphs.append(last_paragraph)
        elif last_paragraph and paragraphs:
            paragraphs[-1] = paragraphs[-1] + " " + last_paragraph
        elif last_paragraph:
            paragraphs.append(last_paragraph)
        
        logger.info(f"Found {len(paragraphs)} paragraphs after boundary detection")
        
        # For summary tasks, try to create fewer, larger chunks
        target_size = self.chunk_size
        if self.is_summary_task:
            # For summaries we want larger chunks even if we exceed the default max
            target_size = min(self.max_single_chunk_size, self.chunk_size * 2)
        
        # If we have too many paragraphs, combine them to reach target chunk count
        if len(paragraphs) > max_chunks:
            logger.info(f"Combining paragraphs to reach target of {max_chunks} chunks (target size: {target_size})")
            
            combined = []
            current = []
            current_size = 0
            
            for para in paragraphs:
                if current_size + len(para) > target_size and current:
                    combined.append(" ".join(current))
                    current = [para]
                    current_size = len(para)
                else:
                    current.append(para)
                    current_size += len(para)
            
            if current:
                combined.append(" ".join(current))
                
            return combined
                
        return paragraphs
    
    def _chunk_by_sentences(self, text: str, chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 200) -> List[str]:
        """
        Split text into chunks by sentences.
        
        Args:
            text: The text to chunk
            chunk_size: Target size of chunks in characters
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum size for a chunk in characters
            
        Returns:
            List of text chunks
        """
        logger.info(f"Sentence chunking with chunk_size={chunk_size}, overlap={overlap}, min_chunk_size={min_chunk_size}")
        
        # Split text into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            logger.warning("No sentences found in text")
            return []
        
        # Log sentence info
        avg_sent_len = sum(len(s) for s in sentences) / max(1, len(sentences))
        logger.info(f"Found {len(sentences)} sentences (avg length: {avg_sent_len:.1f} chars)")
        
        # Create chunks by combining sentences
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content
            if current_size + sentence_len > chunk_size and current_chunk:
                # Only store if the chunk meets the minimum size
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)
                
                # Find sentences that fit within the overlap size
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    s_len = len(s)
                    if overlap_size + s_len <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_len
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 200) -> List[str]:
        """
        Split text into overlapping chunks based on character count.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum size for a chunk in characters
            
        Returns:
            List of text chunks
        """
        logger.info(f"Character chunking with chunk_size={chunk_size}, overlap={overlap}, min_chunk_size={min_chunk_size}")
        
        if not text or not text.strip():
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Define the initial chunk end
            end = min(start + chunk_size, text_len)
            
            # Try to find a natural break point if not at the end of text
            if end < text_len:
                # Look for paragraph break
                next_para = text.find('\n\n', end - overlap, end + 100)
                if next_para != -1:
                    end = next_para + 2
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        next_sent = text.find(punct, end - overlap, end + 100)
                        if next_sent != -1:
                            end = next_sent + len(punct)
                            break
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= min_chunk_size:  # Only add non-empty chunks that meet size minimum
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = max(end - overlap, start + 1)
            
        logger.info(f"Created {len(chunks)} character-based chunks")
        return chunks
    
    def _force_chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Force chunking of text without respecting natural boundaries.
        Use this as a last resort when other methods fail.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            
        Returns:
            List of text chunks
        """
        logger.info("Using forced chunking as fallback")
        chunks = []
        
        # Split text into equal-sized chunks regardless of content
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        
        # Ensure we have at least one chunk
        if not chunks and text.strip():
            chunks = [text.strip()]
            
        logger.info(f"Created {len(chunks)} forced chunks")
        return chunks
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, task_type: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process text directly without a file.
        
        Args:
            text: Text to process
            metadata: Optional metadata
            task_type: Optional task type ('summary', 'qa', etc.)
            
        Returns:
            Tuple of (list of text chunks, list of metadata dictionaries)
        """
        # Check if this is a summary task
        if task_type == "summary":
            self.set_summary_mode(True)
            
        # Preprocess text
        text = self._preprocess_text(text)
        
        # For summary tasks with small documents, use the whole document
        if self.is_summary_task and len(text) <= self.max_single_chunk_size:
            logger.info(f"Summary task with small document ({len(text)} chars), using whole document")
            chunks = [text]
        # For small documents, also consider using the whole document
        elif len(text) <= self.max_single_chunk_size // 2:
            logger.info(f"Small document ({len(text)} chars), using whole document")
            chunks = [text]
        else:
            # Choose chunking strategy
            if self.chunking_mode == "paragraph":
                chunks = self._chunk_by_paragraphs(text, min_chunk_size=self.min_chunk_size)
            elif self.chunking_mode == "sentence":
                chunks = self._chunk_by_sentences(text, self.chunk_size, self.chunk_overlap, min_chunk_size=self.min_chunk_size)
            elif self.chunking_mode == "character":
                chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap, min_chunk_size=self.min_chunk_size)
            elif self.chunking_mode == "whole":
                if len(text) > self.max_single_chunk_size:
                    chunks = self._split_large_document(text)
                else:
                    chunks = [text]
            else:  # "auto" mode
                if len(text) < 10000:
                    chunks = self._chunk_by_paragraphs(text, min_chunk_size=self.min_chunk_size)
                else:
                    chunks = self._chunk_text(text, self.chunk_size, self.chunk_overlap, min_chunk_size=self.min_chunk_size)
        
        # Create base metadata
        base_metadata = {"source": "Direct text"}
        if metadata:
            base_metadata.update(metadata)
        
        # Create chunk-specific metadata
        chunk_metadata = []
        document_id = base_metadata.get("document_id") or str(uuid.uuid4())
        for i, chunk in enumerate(chunks):
            metadata_item = {
                **base_metadata,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "document_id": document_id,
                "chunk_position": f"{i+1}/{len(chunks)}",
                "chunk_size_chars": len(chunk),
                "chunk_size_words": len(chunk.split())
            }
            chunk_metadata.append(metadata_item)
        
        # Reset summary mode
        self.set_summary_mode(False)
        
        return chunks, chunk_metadata
