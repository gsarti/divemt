import re
import sys
from typing import List, Tuple
from xml.dom.minidom import parse

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum


class TercomEdit(StrEnum):
    """
    Tercom error types
    """

    CORRECT = "C"
    SUBSTITUTION = "S"
    INSERTION = "I"
    DELETION = "D"


def parse_tercom_xml_file(filepath):
    """
    parse the xml tree, extracting hypotheses, their references and their edits
    """
    dom = parse(filepath)
    hyp_elems = dom.getElementsByTagName("hyp")
    hyps = []  # list of list of tokens
    refs = []  # list of list of tokens
    hyps_edits = []  # list of list of edits
    for hyp_edits in hyp_elems:
        hyp = []
        ref = []
        h_edits = []
        for edits in hyp_edits.childNodes:
            edit_list = edits.data.split()
            for edit in edit_list:
                trimmed_edit = edit.strip()
                m = re.findall(r'^(".*"),(".*"),(.*),(.*)$', trimmed_edit)
                assert len(m) == 1 and len(m[0]) == 4
                splitted = list(m[0])
                word_ref = splitted[0].strip('"')
                word_hyp = splitted[1].strip('"')
                error_type = splitted[2]
                # print word_hyp, word_ref, error_type
                hyp.append(word_hyp)
                ref.append(word_ref)
                h_edits.append(error_type)
        assert len(hyp) == len(ref)
        hyps.append(hyp)
        refs.append(ref)
        hyps_edits.append(h_edits)
    return hyps, refs, hyps_edits


def align_sentence_tercom(mt_sentence: str, pe_sentence: str, edits: List[str]) -> List[Tuple[int, int]]:
    mt_idx = 0
    pe_idx = 0
    aligns = []
    for edit in edits:
        if edit in [TercomEdit.CORRECT, TercomEdit.SUBSTITUTION]:
            if edit == TercomEdit.CORRECT:
                # Sanity check
                if mt_sentence[mt_idx].lower() != pe_sentence[pe_idx].lower():
                    raise Exception(
                        f"Reading Tercom xml failed, {mt_sentence[mt_idx].lower()} != {pe_sentence[pe_idx].lower()}"
                    )
            aligns.append((mt_idx, pe_idx))
            pe_idx += 1
            mt_idx += 1
        elif edit == TercomEdit.INSERTION:
            aligns.append((None, pe_idx))
            pe_idx += 1
        elif edit == TercomEdit.DELETION:
            aligns.append((mt_idx, None))
            mt_idx += 1
        else:
            raise Exception(f"Unknown edit type {edit}")
    return aligns
