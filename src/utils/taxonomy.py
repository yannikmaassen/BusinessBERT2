from typing import Dict, List, Tuple
import torch


def _row_normalize(m: torch.Tensor) -> torch.Tensor:
    if m.numel() == 0:
        return m
    row_sums = m.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return m / row_sums


def build_taxonomy_maps(rows: List[dict], f2: str, f3: str, f4: str) -> Dict:
    # Filter out "NA" values when building valid SIC code lists
    s2 = sorted({
        str(r.get(f2, "")).strip()
        for r in rows
        if str(r.get(f2, "")).strip() and str(r.get(f2, "")).strip().upper() != "NA"
    })
    s3 = sorted({
        str(r.get(f3, "")).strip()
        for r in rows
        if str(r.get(f3, "")).strip() and str(r.get(f3, "")).strip().upper() != "NA"
    })
    s4 = sorted({
        str(r.get(f4, "")).strip()
        for r in rows
        if str(r.get(f4, "")).strip() and str(r.get(f4, "")).strip().upper() != "NA"
    })

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


def build_label_index_mappings(
    sic4_code_list: List[str],
    sic3_code_list: List[str],
    sic2_code_list: List[str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Create string-to-index dictionaries for each SIC level.
    """
    sic4_code_to_index = {code: index for index, code in enumerate(sorted(sic4_code_list))}
    sic3_code_to_index = {code: index for index, code in enumerate(sorted(sic3_code_list))}
    sic2_code_to_index = {code: index for index, code in enumerate(sorted(sic2_code_list))}

    return sic4_code_to_index, sic3_code_to_index, sic2_code_to_index


def build_child_to_parent_indicator_matrices(
    sic4_to_sic3_mapping: Dict[str, str],
    sic3_to_sic2_mapping: Dict[str, str],
    sic4_code_to_index: Dict[str, int],
    sic3_code_to_index: Dict[str, int],
    sic2_code_to_index: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense child-to-parent indicator matrices:
      child_to_parent_matrix_sic4_to_sic3: shape [number_sic4, number_sic3]
      child_to_parent_matrix_sic4_to_sic2: shape [number_sic4, number_sic2]

    These are used to sum four-digit leaf probabilities upward:
      implied_probabilities_sic3 = probabilities_sic4 @ child_to_parent_matrix_sic4_to_sic3
      implied_probabilities_sic2 = probabilities_sic4 @ child_to_parent_matrix_sic4_to_sic2
    """
    number_of_sic4_classes = len(sic4_code_to_index)
    number_of_sic3_classes = len(sic3_code_to_index)
    number_of_sic2_classes = len(sic2_code_to_index)

    child_to_parent_matrix_sic4_to_sic3 = torch.zeros(
        (number_of_sic4_classes, number_of_sic3_classes), dtype=torch.float32
    )
    child_to_parent_matrix_sic4_to_sic2 = torch.zeros(
        (number_of_sic4_classes, number_of_sic2_classes), dtype=torch.float32
    )

    for sic4_code, sic3_parent_code in sic4_to_sic3_mapping.items():
        sic4_index = sic4_code_to_index[sic4_code]
        sic3_index = sic3_code_to_index[sic3_parent_code]
        child_to_parent_matrix_sic4_to_sic3[sic4_index, sic3_index] = 1.0

        sic2_parent_code = sic3_to_sic2_mapping[sic3_parent_code]
        sic2_index = sic2_code_to_index[sic2_parent_code]
        child_to_parent_matrix_sic4_to_sic2[sic4_index, sic2_index] = 1.0

    return child_to_parent_matrix_sic4_to_sic3, child_to_parent_matrix_sic4_to_sic2