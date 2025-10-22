from typing import List
import random


class PretrainExample:
    def __init__(self,
     #             sentence_a, sentence_b, sop_label,
     sentences, sic2, sic3, sic4):
        # self.sentence_a = sentence_a
        # self.sentence_b = sentence_b
        # self.sop_label = sop_label  # 1 = correct order, 0 = swapped
        self.sentences = sentences
        self.sic2 = sic2
        self.sic3 = sic3
        self.sic4 = sic4


def make_examples(rows: List[dict], field_sentences: str, field_sic2: str, field_sic3: str, field_sic4: str):
    examples: List[PretrainExample] = []
    for row in rows:
        sentences_list = row.get(field_sentences, [])
        if not isinstance(sentences_list, list) or len(sentences_list) == 0:
            continue

        # Join all sentences into one text for MLM
        sentences = ' '.join(sentences_list)
        # if not isinstance(sentences, list) or len(sentences) == 0:
        #     continue
        # sentence_a = sentences[0]
        # sentence_b = " ".join(sentences[1:]) if len(sentences) > 1 else ""
        # sop_label = 1
        # if random.random() < 0.5:
        #     sentence_a, sentence_b = sentence_b, sentence_a
        #     sop_label = 0

        # Handle "NA" values by converting them to empty strings
        sic2_val = str(row.get(field_sic2, "")).strip()
        sic2_val = "" if sic2_val.upper() == "NA" else sic2_val

        sic3_val = str(row.get(field_sic3, "")).strip()
        sic3_val = "" if sic3_val.upper() == "NA" else sic3_val

        sic4_val = str(row.get(field_sic4, "")).strip()
        sic4_val = "" if sic4_val.upper() == "NA" else sic4_val

        examples.append(PretrainExample(
            sentences=sentences,
            sic2=sic2_val,
            sic3=sic3_val,
            sic4=sic4_val,
        ))

    return examples