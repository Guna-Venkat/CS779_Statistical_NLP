import re

def clean_english_hindi_text(line: str) -> str:
    """
    Keep only English (a-zA-Z), Hindi (Devanagari U+0900–U+097F), 
    digits, whitespace, and basic punctuation.
    Remove emojis and other scripts.
    """
    # Allowed ranges: English, Hindi (Devanagari), digits, punctuation, whitespace
    allowed_pattern = re.compile(r'[a-zA-Z\u0900-\u097F0-9\s.,!?;:\'\"-]+')
    
    # Find all valid substrings and join them
    cleaned = ''.join(allowed_pattern.findall(line))
    
    # Normalize spaces
    return re.sub(r'\s+', ' ', cleaned).strip()


def filter_file(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            cleaned_line = clean_english_hindi_text(line)
            if cleaned_line:  # only write non-empty lines
                fout.write(cleaned_line + '\n')


if __name__ == "__main__":
    filter_file("train_large.txt", "train_cleaned.txt")
