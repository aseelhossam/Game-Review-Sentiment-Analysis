import argparse
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from symspellpy import SymSpell, Verbosity
import emoji
import os

# ==========================
# Setup
# ==========================

nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dictionary_path = os.path.join(BASE_DIR, "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

STOPWORDS = set(stopwords.words("english"))

GAME_TERMS = {
    "palworld", "pal", "pals", "jetragon", "dota", "valorant",
    "cs2", "csgo", "counterstrike", "dota2", "blickie", "pal's"
}

# ==========================
# Cleaning Functions
# ==========================

def to_lower(text):
    return text.lower()


def remove_noise(text):
    text = re.sub(r"http\S+", "", text)                    # URLs
    text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)     # HTML tags
    text = re.sub(r"\[.*?\]", "", text, flags=re.DOTALL)   # Steam markup tags (multiline-safe)
    text = re.sub(r"\s+", " ", text)                       # Extra whitespace
    return text.strip()


def handle_emojis(text):
    text = emoji.demojize(text)                            # Convert emojis to :text:
    text = text.replace(":", " ").replace("_", " ")        # Clean delimiters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)            # Remove remaining non-ASCII (e.g. Steam ♥♥♥♥)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_repeated_chars(text):
    """Collapse 3+ repeated characters to 2: 'amazinnngg' → 'amazingg'"""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def fix_spelling(text):
    corrected_words = []
    for word in text.split():
        stripped = word.strip(string.punctuation)
        punct_suffix = word[len(stripped):]               # Preserve trailing punctuation

        if not stripped or stripped.lower() in GAME_TERMS:
            corrected_words.append(word)
            continue

        suggestions = sym_spell.lookup(stripped, Verbosity.CLOSEST, max_edit_distance=2)
        corrected = suggestions[0].term if suggestions else stripped
        corrected_words.append(corrected + punct_suffix)

    return " ".join(corrected_words)


def remove_stopwords(text):
    return " ".join(w for w in text.split() if w.lower() not in STOPWORDS)


def lemmatize_text(text):
    """POS-aware lemmatization: verbs and adjectives lemmatized correctly."""
    pos_map = {
        "VB": "v", "VBD": "v", "VBG": "v",
        "VBN": "v", "VBP": "v", "VBZ": "v",
        "JJ": "a", "JJR": "a", "JJS": "a"
    }
    try:
        blob = TextBlob(text)
        return " ".join(
            Word(word).lemmatize(pos_map.get(tag, "n"))
            for word, tag in blob.tags
        )
    except Exception:
        # Fallback to noun-only lemmatization if TextBlob POS tagging fails
        return " ".join(Word(w).lemmatize() for w in text.split())


def extract_category(df):
    mapping = {
        "palworld": "rpg_survival",
        "dota 2": "moba",
        "counter-strike 2": "fps"
    }
    df["category"] = df["game_name"].str.lower().map(mapping)
    return df


# ==========================
# Main Function
# ==========================

def main(args):
    df = pd.read_csv(args.input)

    if "review_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'review_text' column")

    # Ensure all review_text values are strings (guard against NaN)
    df["review_text"] = df["review_text"].fillna("").astype(str)

    # --- Pipeline (order matters) ---

    if args.remove_noise:
        df["review_text"] = df["review_text"].apply(remove_noise)

    if args.handle_emojis:
        df["review_text"] = df["review_text"].apply(handle_emojis)

    if args.lower:
        df["review_text"] = df["review_text"].apply(to_lower)

    if args.normalize_chars:
        df["review_text"] = df["review_text"].apply(normalize_repeated_chars)

    if args.fix_spelling:
        df["review_text"] = df["review_text"].apply(fix_spelling)

    if args.remove_stopwords:
        df["review_text"] = df["review_text"].apply(remove_stopwords)

    if args.lemmatize:
        df["review_text"] = df["review_text"].apply(lemmatize_text)

    if args.extract_tags:
        df = extract_category(df)

    df.to_csv(args.output, index=False)
    print(f"Cleaned data saved to {args.output}")


# ==========================
# CLI Arguments
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")

    parser.add_argument("--input",            type=str, required=True,  help="Input CSV file")
    parser.add_argument("--output",           type=str, required=True,  help="Output CSV file")

    parser.add_argument("--remove_noise",     action="store_true", help="Remove URLs, HTML, and Steam markup tags")
    parser.add_argument("--handle_emojis",    action="store_true", help="Convert emojis to text and strip remaining non-ASCII")
    parser.add_argument("--lower",            action="store_true", help="Convert text to lowercase")
    parser.add_argument("--normalize_chars",  action="store_true", help="Collapse 3+ repeated characters to 2 (e.g. 'goood' → 'good')")
    parser.add_argument("--fix_spelling",     action="store_true", help="Apply spell correction (preserves game terms)")
    parser.add_argument("--remove_stopwords", action="store_true", help="Remove common English stopwords")
    parser.add_argument("--lemmatize",        action="store_true", help="Apply POS-aware lemmatization")
    parser.add_argument("--extract_tags",     action="store_true", help="Add game category column")

    args = parser.parse_args()
    main(args)


# ==========================
# Preprocessing Schemes
# ==========================
# Use these when running Phase 4 ML experiments:
#
# Scheme 1 — Light:
#   python main.py --input data.csv --output s1.csv \
#     --remove_noise --handle_emojis --lower
#
# Scheme 2 — Moderate:
#   python main.py --input data.csv --output s2.csv \
#     --remove_noise --handle_emojis --lower --remove_stopwords --lemmatize
#
# Scheme 3 — Aggressive:
#   python main.py --input data.csv --output s3.csv \
#     --remove_noise --handle_emojis --lower \
#     --normalize_chars --fix_spelling --remove_stopwords --lemmatize