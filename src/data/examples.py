from typing import List
import random


class PretrainExample:
    def __init__(self, text_a, text_b, sop_label, sic2, sic3, sic4):
        self.text_a = text_a
        self.text_b = text_b
        self.sop_label = sop_label  # 1 = correct order, 0 = swapped
        self.sic2 = sic2
        self.sic3 = sic3
        self.sic4 = sic4


def make_examples(rows: List[dict], field_sentences: str, field_sic2: str, field_sic3: str, field_sic4: str):
    exs: List[PretrainExample] = []
    for r in rows:
        sents = r.get(field_sentences, [])
        if not isinstance(sents, list) or len(sents) == 0:
            continue
        a = sents[0]
        b = " ".join(sents[1:]) if len(sents) > 1 else ""
        label = 1
        if random.random() < 0.5:
            a, b = b, a
            label = 0
        exs.append(PretrainExample(
            text_a=a, text_b=b, sop_label=label,
            sic2=str(r.get(field_sic2, "")).strip(),
            sic3=str(r.get(field_sic3, "")).strip(),
            sic4=str(r.get(field_sic4, "")).strip(),
        ))
    return exs