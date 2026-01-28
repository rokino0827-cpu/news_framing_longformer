"""
Text processing utilities for handling long documents.
"""
from typing import Optional, List, Dict
from transformers import PreTrainedTokenizer


def join_title_content(title: Optional[str], content: str, use_title: bool) -> str:
    """
    Join title and content with appropriate separator.
    
    Args:
        title: Article title (can be None or empty)
        content: Article content
        use_title: Whether to include title
        
    Returns:
        Combined text string
    """
    if not use_title or not title or title.strip() == "":
        return content.strip()
    
    title = title.strip()
    content = content.strip()
    
    # Add separator between title and content
    if title and content:
        return f"{title} [SEP] {content}"
    elif title:
        return title
    else:
        return content


def truncate_head_tail_tokens(
    input_ids: List[int], 
    max_length: int, 
    tokenizer: PreTrainedTokenizer
) -> List[int]:
    """
    Truncate tokens by keeping head and tail portions.
    Preserves CLS and SEP token structure.
    
    Args:
        input_ids: List of token IDs
        max_length: Maximum sequence length
        tokenizer: Tokenizer for special tokens
        
    Returns:
        Truncated token list
    """
    if len(input_ids) <= max_length:
        return input_ids
    
    # Reserve space for special tokens
    cls_token = tokenizer.cls_token_id
    sep_token = tokenizer.sep_token_id
    
    # Find positions of special tokens
    cls_pos = 0 if input_ids[0] == cls_token else None
    sep_positions = [i for i, token_id in enumerate(input_ids) if token_id == sep_token]
    
    # Calculate how many content tokens we can keep
    special_tokens_count = 1 if cls_pos is not None else 0
    special_tokens_count += len(sep_positions)
    content_budget = max_length - special_tokens_count
    
    if content_budget <= 0:
        # Fallback: just keep special tokens
        result = []
        if cls_pos is not None:
            result.append(cls_token)
        if sep_positions:
            result.extend([sep_token] * min(len(sep_positions), max_length - len(result)))
        return result[:max_length]
    
    # Extract content tokens (excluding special tokens)
    content_tokens = []
    for i, token_id in enumerate(input_ids):
        if i != cls_pos and token_id != sep_token:
            content_tokens.append((i, token_id))
    
    if not content_tokens:
        # No content tokens, just return special tokens
        result = []
        if cls_pos is not None:
            result.append(cls_token)
        result.extend([sep_token] * min(len(sep_positions), max_length - len(result)))
        return result[:max_length]
    
    # Take head and tail portions
    head_count = content_budget // 2
    tail_count = content_budget - head_count
    
    selected_content = []
    if head_count > 0:
        selected_content.extend(content_tokens[:head_count])
    if tail_count > 0 and len(content_tokens) > head_count:
        selected_content.extend(content_tokens[-tail_count:])
    
    # Reconstruct sequence
    result = []
    if cls_pos is not None:
        result.append(cls_token)
    
    # Add content tokens
    for _, token_id in selected_content:
        result.append(token_id)
    
    # Add SEP tokens
    for _ in sep_positions:
        if len(result) < max_length:
            result.append(sep_token)
    
    return result[:max_length]


def sliding_window_encode(
    text: str, 
    tokenizer: PreTrainedTokenizer, 
    window: int, 
    stride: int
) -> Dict:
    """
    Encode text using sliding window approach.
    
    Args:
        text: Input text
        tokenizer: Tokenizer
        window: Window size
        stride: Stride size
        
    Returns:
        Dictionary with encoded chunks
    """
    # Tokenize full text
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + window]
        if len(chunk_tokens) < window // 4:  # Skip very short chunks
            break
            
        # Convert back to text and encode
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        encoding = tokenizer(
            chunk_text,
            max_length=window,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        chunks.append(encoding)
    
    return {
        'chunks': chunks,
        'num_chunks': len(chunks)
    }