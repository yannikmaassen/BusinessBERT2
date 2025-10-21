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


def create_sentence_pairs(sentences, max_sentences_per_segment=8):
    """
    Creates sentence pairs for both SOP and MLM objectives from a document.

    For each document:
    1. Creates natural consecutive sentence pairs (preserves SOP objective)
    2. Ensures each pair is of manageable length
    3. Uses a sliding window approach to maximize context coverage

    Args:
        sentences: List of sentences from the document
        max_sentences_per_segment: Maximum number of sentences to combine in a single segment

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

    # For longer documents, create meaningful segments
    # Use a sliding window approach with 50% overlap to maximize coverage
    stride = max(1, max_sentences_per_segment // 2)

    for i in range(0, len(sentences) - 1, stride):
        # Define the boundary between sentence_a and sentence_b
        # Ensure we have content for both parts
        mid_point = min(i + max_sentences_per_segment // 2, len(sentences) - 1)
        end_point = min(i + max_sentences_per_segment, len(sentences))

        # Skip if we can't form a proper pair
        if mid_point <= i:
            continue

        # Create meaningful segments
        sentence_a = " ".join(sentences[i:mid_point])
        sentence_b = " ".join(sentences[mid_point:end_point])

        # Only add if both segments have content
        if sentence_a and sentence_b:
            pairs.append((sentence_a, sentence_b))

    # If we couldn't create any pairs, try a simpler approach
    if not pairs and len(sentences) > 1:
        # Split document in half
        mid = len(sentences) // 2
        return [(
            " ".join(sentences[:mid]),
            " ".join(sentences[mid:])
        )]

    return pairs


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

        # Generate balanced sentence pairs for both MLM and SOP objectives
        sentence_pairs = create_sentence_pairs(sentences)

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