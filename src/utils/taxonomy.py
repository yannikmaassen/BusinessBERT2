from typing import Dict, List, Tuple, Optional
import torch


def _row_normalize(m: torch.Tensor) -> torch.Tensor:
    if m.numel() == 0:
        return m
    row_sums = m.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return m / row_sums


def _normalize_code(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() == "NA":
        return None
    return s


def build_code_index_mappings(
    sic2_code_list: List[str],
    sic3_code_list: List[str],
    sic4_code_list: List[str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    sic2_code_to_index = {code: index for index, code in enumerate(sorted(sic2_code_list))}
    sic3_code_to_index = {code: index for index, code in enumerate(sorted(sic3_code_list))}
    sic4_code_to_index = {code: index for index, code in enumerate(sorted(sic4_code_list))}

    return sic2_code_to_index, sic3_code_to_index, sic4_code_to_index


def build_taxonomy_maps(rows: List[dict], field_sic2: str, field_sic3: str, field_sic4: str) -> Dict:
    sic2_set = set()
    sic3_set = set()
    sic4_set = set()

    for r in rows:
        c2 = _normalize_code(r.get(field_sic2))
        c3 = _normalize_code(r.get(field_sic3))
        c4 = _normalize_code(r.get(field_sic4))
        if c2 is not None:
            sic2_set.add(c2)
        if c3 is not None:
            sic3_set.add(c3)
        if c4 is not None:
            sic4_set.add(c4)

    sic2_list = sorted(sic2_set)
    sic3_list = sorted(sic3_set)
    sic4_list = sorted(sic4_set)

    indexed_sic2_list, indexed_sic3_list, indexed_sic4_list = (
        build_code_index_mappings(sic2_list, sic3_list, sic4_list)
    )

    parent3_to2: Dict[int, Optional[int]] = {}
    for sic3 in sic3_list:
        i3 = indexed_sic3_list[sic3]
        parent_code = sic3[:2]
        i2 = indexed_sic2_list.get(parent_code, None)
        parent3_to2[i3] = i2

    parent4_to3: Dict[int, Optional[int]] = {}
    for sic4 in sic4_list:
        i4 = indexed_sic4_list[sic4]
        parent_code = sic4[:3]
        i3 = indexed_sic3_list.get(parent_code, None)
        parent4_to3[i4] = i3

    A32 = torch.zeros((len(sic3_list), len(sic2_list)), dtype=torch.float32)
    for i3, i2 in parent3_to2.items():
        if i2 is not None:
            A32[i3, i2] = 1.0
    A32 = A32.contiguous()

    A43 = torch.zeros((len(sic4_list), len(sic3_list)), dtype=torch.float32)
    for i4, i3 in parent4_to3.items():
        if i3 is not None:
            A43[i4, i3] = 1.0
    A43 = A43.contiguous()

    return {
        "sic2_list": sic2_list,
        "sic3_list": sic3_list,
        "sic4_list": sic4_list,
        "idx2": indexed_sic2_list,
        "idx3": indexed_sic3_list,
        "idx4": indexed_sic4_list,
        "parent3_to2": parent3_to2,
        "parent4_to3": parent4_to3,
        "A32": A32,
        "A43": A43,
    }
