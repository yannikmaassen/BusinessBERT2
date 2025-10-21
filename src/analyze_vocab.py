import argparse
import collections
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from src.utils.file_manager import read_jsonl

def analyze_tokenization(texts: List[str], tokenizer, field_name):
    """Analyze tokenization patterns and identify potentially problematic tokens."""
    # Calculate token frequencies
    token_counter = collections.Counter()
    unknown_counter = collections.Counter()
    seq_lengths = []

    for text in texts:
        if not text:
            continue

        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        seq_lengths.append(len(tokens))

        # Count tokens
        token_counter.update(tokens)

        # Track unknown tokens
        if tokenizer.unk_token in tokens:
            words = text.split()
            encoded = tokenizer.encode(text, add_special_tokens=False)
            unknown_indices = [i for i, id in enumerate(encoded) if id == tokenizer.unk_token_id]

            for idx in unknown_indices:
                if idx < len(words):
                    unknown_counter.update([words[idx]])

    # Calculate statistics
    stats = {
        "total_texts": len(texts),
        "total_tokens": sum(seq_lengths),
        "unique_tokens": len(token_counter),
        "avg_sequence_length": np.mean(seq_lengths),
        "max_sequence_length": max(seq_lengths),
        "unknown_token_count": token_counter.get(tokenizer.unk_token, 0),
        "unknown_token_percent": token_counter.get(tokenizer.unk_token, 0) / sum(token_counter.values()) * 100,
        "top_unknown_words": unknown_counter.most_common(20),
        "top_tokens": token_counter.most_common(20)
    }

    print(f"\n===== Tokenization Analysis for {field_name} =====")
    print(f"Total texts analyzed: {stats['total_texts']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Unique tokens used: {stats['unique_tokens']}")
    print(f"Average sequence length: {stats['avg_sequence_length']:.2f} tokens")
    print(f"Maximum sequence length: {stats['max_sequence_length']} tokens")
    print(f"Unknown token count: {stats['unknown_token_count']} ({stats['unknown_token_percent']:.2f}%)")

    print("\nTop 20 tokens:")
    for token, count in stats['top_tokens']:
        print(f"  {token}: {count}")

    if stats['unknown_token_count'] > 0:
        print("\nTop unknown words:")
        for word, count in stats['top_unknown_words']:
            print(f"  {word}: {count}")

    # Plot token frequency distribution
    plt.figure(figsize=(12, 6))

    # Get top 100 tokens
    top_tokens = token_counter.most_common(100)
    tokens, counts = zip(*top_tokens)

    plt.bar(range(len(tokens)), counts)
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.title(f"Top 100 Token Frequencies in {field_name}")
    plt.tight_layout()
    plt.savefig(f"token_freq_{field_name.replace(' ', '_')}.png")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Analyze vocabulary and tokenization patterns in the dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to the JSONL dataset")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.data}")
    dataset = read_jsonl(args.data)
    print(f"Loaded {len(dataset)} rows")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    field = "sentences"

    # Extract texts
    all_texts = []
    for row in dataset:
        sentences = row.get(field, [])
        if not isinstance(sentences, list):
            continue

        # Join all sentences for this example
        text = " ".join(sentences)
        if text.strip():
            all_texts.append(text)

    print(f"Analyzing {len(all_texts)} text examples")
    stats = analyze_tokenization(all_texts, tokenizer)

if __name__ == "__main__":
    main()
