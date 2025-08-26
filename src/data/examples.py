from typing import List
import random


# TODO: IC
class PretrainExample:
    def __init__(self, sentence_a, sentence_b, sop_label, sic2, sic3, sic4):
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b
        self.sop_label = sop_label  # 1 = correct order, 0 = swapped
        self.sic2 = sic2
        self.sic3 = sic3
        self.sic4 = sic4


def make_examples(rows: List[dict], field_sentences: str, field_sic2: str, field_sic3: str, field_sic4: str):
    examples: List[PretrainExample] = []
    for row in rows:
        sentences = row.get(field_sentences, [])
        if not isinstance(sentences, list) or len(sentences) == 0:
            continue
        sentence_a = sentences[0]
        sentence_b = " ".join(sentences[1:]) if len(sentences) > 1 else ""
        sop_label = 1
        if random.random() < 0.5:
            sentence_a, sentence_b = sentence_b, sentence_a
            sop_label = 0
        examples.append(PretrainExample(
            sentence_a=sentence_a, sentence_b=sentence_b, sop_label=sop_label,
            sic2=str(row.get(field_sic2, "")).strip(),
            sic3=str(row.get(field_sic3, "")).strip(),
            sic4=str(row.get(field_sic4, "")).strip(),
        ))
    return examples