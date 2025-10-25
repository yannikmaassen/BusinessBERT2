from typing import Dict, List, Tuple
import torch


def _row_normalize(m: torch.Tensor) -> torch.Tensor:
    if m.numel() == 0:
        return m
    row_sums = m.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return m / row_sums


def build_taxonomy_maps(rows: List[dict], field_sic2: str, field_sic3: str, field_sic4: str) -> Dict:
    # Raw value examples:
        # - sic2: ["20", "28", "35", "50", "73"]
        # - sic3: ["201", "282", "357", "501", "737"]
        # - sic4: ["2011", "2821", "3571", "5012", "7371"]
    # Filter out "NA" values when building valid SIC code lists
    sic2_list = sorted({
        str(row.get(field_sic2, "")).strip()
        for row in rows
        if str(row.get(field_sic2, "")).strip() and str(row.get(field_sic2, "")).strip().upper() != "NA"
    })

    sic3_list = sorted({
        str(row.get(field_sic3, "")).strip()
        for row in rows
        if str(row.get(field_sic3, "")).strip() and str(row.get(field_sic3, "")).strip().upper() != "NA"
    })

    sic4_list = sorted({
        str(row.get(field_sic4, "")).strip()
        for row in rows
        if str(row.get(field_sic4, "")).strip() and str(row.get(field_sic4, "")).strip().upper() != "NA"
    })

    indexed_sic2_list, indexed_sic3_list, indexed_sic4_list = (
        build_code_index_mappings(sic2_list, sic3_list, sic4_list))

    # After sorting and indexing:
        # - indexed_sic2_list = { "20": 0, "28": 1, "35": 2, "50": 3, "73": 4}
        # - indexed_sic3_list = { "201": 0, "282": 1, "357": 2, "501": 3, "737": 4}
        # - indexed_sic4_list = { "2011": 0, "2821": 1, "3571": 2, "5012": 3, "7371": 4}

    # Hierarchical parent mappings
    parent3_to2 = {}
    for sic3 in sic3_list:
        parent3_to2[indexed_sic3_list[sic3]] = indexed_sic3_list.get(sic3[:2], None)

    parent4_to3 = {}
    for sic4 in sic4_list:
        parent4_to3[indexed_sic4_list[sic4]] = indexed_sic4_list.get(sic4[:3], None)

    # Indicator matrices
        # A32: shape [num_sic3 × num_sic2]
        # Each row is a SIC3 code, each column is a SIC2 code
        # A32[i3, i2] = 1 if SIC3(i3) has parent SIC2(i2)
    A32 = torch.zeros((len(sic3_list), len(sic2_list)))
    for i3, i2 in parent3_to2.items():
        if i2 is not None:
            A32[i3, i2] = 1.0

        # A43: shape [num_sic4 × num_sic3]
        # Each row is a SIC4 code, each column is a SIC3 code
        # A43[i4, i3] = 1 if SIC4(i4) has parent SIC3(i3)
    A43 = torch.zeros((len(sic4_list), len(sic3_list)))
    for i4, i3 in parent4_to3.items():
        if i3 is not None:
            A43[i4, i3] = 1.0

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


def build_code_index_mappings(
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