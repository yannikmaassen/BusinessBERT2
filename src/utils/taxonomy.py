from typing import Dict, List
import torch


def _row_normalize(m: torch.Tensor) -> torch.Tensor:
    if m.numel() == 0:
        return m
    row_sums = m.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return m / row_sums


def build_taxonomy_maps(rows: List[dict], f2: str, f3: str, f4: str) -> Dict:
    s2 = sorted({str(r.get(f2, "")).strip() for r in rows if str(r.get(f2, "")).strip()})
    s3 = sorted({str(r.get(f3, "")).strip() for r in rows if str(r.get(f3, "")).strip()})
    s4 = sorted({str(r.get(f4, "")).strip() for r in rows if str(r.get(f4, "")).strip()})

    idx2 = {c: i for i, c in enumerate(s2)}
    idx3 = {c: i for i, c in enumerate(s3)}
    idx4 = {c: i for i, c in enumerate(s4)}

    parent3_to2 = {}
    for c3 in s3:
        parent3_to2[idx3[c3]] = idx2.get(c3[:2], None)

    parent4_to3 = {}
    for c4 in s4:
        parent4_to3[idx4[c4]] = idx3.get(c4[:3], None)

    A32 = torch.zeros((len(s3), len(s2)))
    for i3, i2 in parent3_to2.items():
        if i2 is not None:
            A32[i3, i2] = 1.0

    A43 = torch.zeros((len(s4), len(s3)))
    for i4, i3 in parent4_to3.items():
        if i3 is not None:
            A43[i4, i3] = 1.0

    B23 = _row_normalize(A32.T.clone()) if A32.numel() else A32.T.clone()
    B34 = _row_normalize(A43.T.clone()) if A43.numel() else A43.T.clone()

    return {
        "sic2_list": s2, "sic3_list": s3, "sic4_list": s4,
        "idx2": idx2, "idx3": idx3, "idx4": idx4,
        "parent3_to2": parent3_to2, "parent4_to3": parent4_to3,
        "A32": A32, "A43": A43,
        "B23": B23, "B34": B34,
    }