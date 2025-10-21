from typing import List
import random


class PretrainExample:
    def __init__(self, sentence_a, sentence_b, sop_label, sic2, sic3, sic4):
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b
        self.sop_label = sop_label  # 1 = correct order, 0 = swapped
        self.sic2 = sic2
        self.sic3 = sic3
        self.sic4 = sic4


def create_memory_efficient_pairs(sentences, max_pairs_per_doc=3, max_sentences_per_segment=6):
    """
    Creates a limited number of sentence pairs from a document to control memory usage.

    Strategy:
    1. For short documents: Use the entire document
    2. For long documents: Create a small fixed number of well-distributed segments

    Args:
        sentences: List of sentences from the document
        max_pairs_per_doc: Maximum number of pairs to create from one document
        max_sentences_per_segment: Maximum sentences to include in a single segment

    Returns:
        List of tuples (sentence_a, sentence_b) ready for SOP
    """
    pairs = []

    # Handle short documents (1-2 sentences)
    if len(sentences) <= 2:
        if len(sentences) == 1:
            return [(sentences[0], "")]
        else:
            return [(sentences[0], sentences[1])]

    # For longer documents, create a limited number of strategic samples
    doc_length = len(sentences)

    # If document is not too long, we can simply split it once
    if doc_length <= max_sentences_per_segment * 2:
        mid = doc_length // 2
        return [(
            " ".join(sentences[:mid]),
            " ".join(sentences[mid:])
        )]

    # For very long documents, sample from beginning, middle and end
    # This ensures coverage of the document without creating too many examples
    samples = []

    # Calculate segment size based on document length
    segment_size = min(max_sentences_per_segment, (doc_length // (max_pairs_per_doc * 2)))

    # Beginning of document
    samples.append((
        " ".join(sentences[:segment_size]),
        " ".join(sentences[segment_size:segment_size*2])
    ))

    # Middle of document (if long enough)
    if doc_length > max_sentences_per_segment * 4:
        mid_point = doc_length // 2
        samples.append((
            " ".join(sentences[mid_point-segment_size:mid_point]),
            " ".join(sentences[mid_point:mid_point+segment_size])
        ))

    # End of document
    if doc_length > max_sentences_per_segment * 2:
        samples.append((
            " ".join(sentences[-(segment_size*2):-segment_size]),
            " ".join(sentences[-segment_size:])
        ))

    # Limit the number of samples to control memory usage
    return samples[:max_pairs_per_doc]


def make_examples(rows: List[dict], field_sentences: str, field_sic2: str, field_sic3: str, field_sic4: str):
    examples: List[PretrainExample] = []
    for row in rows:
        sentences = row.get(field_sentences, [])
        if not isinstance(sentences, list) or len(sentences) == 0:
            continue

        # Handle "NA" values by converting them to empty strings
        sic2_val = str(row.get(field_sic2, "")).strip()
        sic2_val = "" if sic2_val.upper() == "NA" else sic2_val

        sic3_val = str(row.get(field_sic3, "")).strip()
        sic3_val = "" if sic3_val.upper() == "NA" else sic3_val

        sic4_val = str(row.get(field_sic4, "")).strip()
        sic4_val = "" if sic4_val.upper() == "NA" else sic4_val

        # Generate a limited number of memory-efficient pairs
        sentence_pairs = create_memory_efficient_pairs(sentences)

        for sentence_a, sentence_b in sentence_pairs:
            # Skip if either part is empty
            if not sentence_a:
                continue

            # Apply SOP (sentence order prediction) with 50% chance of swapping
            sop_label = 1  # 1 = correct order
            if random.random() < 0.5:
                sentence_a, sentence_b = sentence_b, sentence_a
                sop_label = 0  # 0 = swapped order

            examples.append(PretrainExample(
                sentence_a=sentence_a,
                sentence_b=sentence_b,
                sop_label=sop_label,
                sic2=sic2_val,
                sic3=sic3_val,
                sic4=sic4_val,
            ))

    return examples