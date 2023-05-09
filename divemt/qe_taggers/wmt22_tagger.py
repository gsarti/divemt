import codecs
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union
from xml.sax.saxutils import escape

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum
from simalign import SentenceAligner
from tqdm import tqdm

from ..parse_utils import clear_nlp_cache
from .base import QETagger, TAlignment, TTag
from .wmt22qe_utils import align_sentence_tercom, parse_tercom_xml_file

logger = logging.getLogger(__name__)


class FluencyRule(StrEnum):
    """Fluency rules used in the WMT22 QE task."""

    NORMAL = "normal"
    MISSING = "missing-only"
    IGNORE_SHF = "ignore-shift-set"


class OmissionRule(StrEnum):
    """Omission rules used in the WMT22 QE task."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


class WMT22QETags(StrEnum):
    """WMT22 QE tags"""

    OK = "OK"
    BAD = "BAD"


class WMT22QETagger(QETagger):
    """Mimics the word-level QE tagging process used for WMT22."""

    ID = "wmt22_qe"

    def __init__(
        self,
        aligner: Optional[SentenceAligner] = None,
        tmp_dir: Optional[str] = None,
        tercom_out: Optional[str] = None,
        tercom_path: Optional[str] = None,
    ):
        """Initialize the WMT22QETagger."""
        self.aligner = aligner if aligner else SentenceAligner(model="xlmr", token_type="bpe", matching_methods="mai")
        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tercom_out = Path(tercom_out) if tercom_out is not None else self.tmp_dir / "tercom"
        self.tercom_path = tercom_path if tercom_path is not None else "scripts/tercom.7.25.jar"

    def align_source_pe(
        self,
        src_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        pe_langs: List[str],
    ) -> List[List[TAlignment]]:
        return [
            self.aligner.get_word_aligns(src_tok, mt_tok)["itermax" if mt_lang not in ["de", "cs"] else "inter"]
            for src_tok, mt_tok, mt_lang in tqdm(
                zip(src_tokens, pe_tokens, pe_langs),
                total=len(src_tokens),
                desc="Aligning src-pe",
            )
        ]

    def align_mt_pe(
        self,
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
    ) -> List[List[TAlignment]]:
        ref_fname = self.tmp_dir / "ref.txt"
        hyp_fname = self.tmp_dir / "hyp.txt"
        # Adapted from https://github.com/deep-spin/qe-corpus-builder/corpus_generation/tools/format_tercom.py
        with codecs.open(str(ref_fname), "w", encoding="utf-8") as rf:
            with codecs.open(str(hyp_fname), "w", encoding="utf-8") as hf:
                for idx, (ref, hyp) in enumerate(zip(mt_tokens, pe_tokens)):
                    _ref = " ".join(ref).rstrip()
                    _ref = escape(_ref).replace('"', '\\"')
                    rf.write(f"{_ref}\t({idx})\n")
                    _hyp = " ".join(hyp).rstrip()
                    _hyp = escape(_hyp).replace('"', '\\"')
                    hf.write(f"{_hyp}\t({idx})\n")
        ps = [
            "java",
            "-jar",
            self.tercom_path,
            "-r",
            ref_fname,
            "-h",
            hyp_fname,
            "-n",
            self.tercom_out,
            "-d",
            "0",
        ]
        try:
            _ = subprocess.run(ps, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Error while running tercom: {e.stderr}.\nPlease make sure you have java installed and that the .jar "
                f"file is found at {self.tercom_path}"
            )
        # Parse tercom HTML
        pe_parse_tokens, mt_parse_tokens, edits = parse_tercom_xml_file(f"{self.tercom_out}.xml")

        # Sanity check: Original and tercom files match in number of tokens
        # Note that we will not use the tokenized tercom outputs only the alignments
        for mt_par_toks, pe_par_toks, mt_toks, pe_toks in zip(mt_parse_tokens, pe_parse_tokens, mt_tokens, pe_tokens):
            # Inserted tokens correspond to empty strings in the XLM tercom output
            assert len([t for t in mt_par_toks if t]) == len(mt_toks), f"{mt_par_toks} != {mt_toks}"
            assert len([t for t in pe_par_toks if t]) == len(pe_toks), f"{pe_par_toks} != {pe_toks}"

        return [align_sentence_tercom(mt, pe, edit) for mt, pe, edit in zip(mt_tokens, pe_tokens, edits)]

    @staticmethod
    def tags_from_edits(
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        alignments: List[List[TAlignment]],
        use_gaps: bool = False,
        omissions: str = OmissionRule.RIGHT.value,
    ) -> List[List[TTag]]:
        """Produce tags on MT tokens from edits found in the PE tokens."""
        if use_gaps:
            omissions = OmissionRule.NONE.value

        mt_tags = []
        for mt_tok, pe_tok, align in tqdm(
            zip(mt_tokens, pe_tokens, alignments),
            desc="Tagging MT",
            total=len(mt_tokens),
        ):
            sent_tags = []
            sent_deletion_indices = []
            mt_position = 0

            # Loop over alignments. This has the length of the edit-distance aligned sequences.
            for mt_idx, pe_idx in align:
                if mt_idx is None:
                    # Deleted word error (need to store for later)
                    if omissions == OmissionRule.LEFT or omissions == OmissionRule.NONE:
                        sent_deletion_indices.append(mt_position - 1)
                    else:
                        sent_deletion_indices.append(mt_position)
                elif pe_idx is None:
                    # Insertion error
                    sent_tags.append(WMT22QETags.BAD.value)
                    mt_position += 1
                elif mt_tok[mt_idx] != pe_tok[pe_idx]:
                    # Substitution error
                    sent_tags.append(WMT22QETags.BAD.value)
                    mt_position += 1
                else:
                    # OK
                    sent_tags.append(WMT22QETags.OK.value)
                    mt_position += 1

            # Insert deletion errors as gaps
            word_and_gaps_tags = []
            if use_gaps:
                # Add starting OK/BAD
                if -1 in sent_deletion_indices:
                    word_and_gaps_tags.append(WMT22QETags.BAD.value)
                else:
                    word_and_gaps_tags.append(WMT22QETags.OK.value)
                # Add rest of OK/BADs
                for index, tag in enumerate(sent_tags):
                    if index in sent_deletion_indices:
                        word_and_gaps_tags.extend([tag, WMT22QETags.BAD.value])
                    else:
                        word_and_gaps_tags.extend([tag, WMT22QETags.OK.value])
                mt_tags.append(word_and_gaps_tags)
            else:
                if omissions == OmissionRule.NONE:
                    mt_tags.append(sent_tags)
                elif omissions == OmissionRule.RIGHT:
                    for index, tag in enumerate(sent_tags):
                        if index in sent_deletion_indices:
                            word_and_gaps_tags.append(WMT22QETags.BAD.value)
                        else:
                            word_and_gaps_tags.append(tag)
                    if len(sent_tags) in sent_deletion_indices:
                        word_and_gaps_tags.append(WMT22QETags.BAD.value)
                    else:
                        word_and_gaps_tags.append(WMT22QETags.OK.value)
                elif omissions == OmissionRule.LEFT:
                    if -1 in sent_deletion_indices:
                        word_and_gaps_tags.append(WMT22QETags.BAD.value)
                    else:
                        word_and_gaps_tags.append(WMT22QETags.OK.value)
                    for index, tag in enumerate(sent_tags):
                        if index in sent_deletion_indices:
                            word_and_gaps_tags.append(WMT22QETags.BAD.value)
                        else:
                            word_and_gaps_tags.append(tag)
                mt_tags.append(word_and_gaps_tags)

        # Basic sanity checks
        if use_gaps:
            assert all(len(aa) * 2 + 1 == len(bb) for aa, bb in zip(mt_tokens, mt_tags)), "MT tag creation failed"
        else:
            if omissions == OmissionRule.NONE:  # noqa: PLR5501
                assert all(len(aa) == len(bb) for aa, bb in zip(mt_tokens, mt_tags)), "MT tag creation failed"
            else:
                assert all(len(aa) + 1 == len(bb) for aa, bb in zip(mt_tokens, mt_tags)), "MT tag creation failed"
        return mt_tags

    @staticmethod
    def tags_to_source(
        src_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        src_pe_alignments: List[List[TAlignment]],
        mt_pe_alignments: List[List[TAlignment]],
        fluency_rule: str = FluencyRule.NORMAL.value,
    ) -> List[List[TTag]]:
        """Propagate tags from MT to source."""
        # Reorganize source-target alignments as a dict
        pe2source = []
        for sent in src_pe_alignments:
            pe2source_sent = defaultdict(list)
            for src_idx, pe_idx in sent:
                pe2source_sent[pe_idx].append(src_idx)
            pe2source.append(pe2source_sent)

        src_tags = []
        for (
            src_sent_tok,
            mt_sent_tok,
            pe_sent_tok,
            sent_pe2src,
            sent_mt_pe_aligns,
        ) in tqdm(
            zip(src_tokens, mt_tokens, pe_tokens, pe2source, mt_pe_alignments),
            desc="Tagging source",
            total=len(src_tokens),
        ):
            source_sentence_bad_indices = set()
            mt_position = 0
            for mt_idx, pe_idx in sent_mt_pe_aligns:
                if mt_idx is None or (
                    mt_idx is not None and pe_idx is not None and mt_sent_tok[mt_idx] != pe_sent_tok[pe_idx]
                ):
                    if fluency_rule == FluencyRule.NORMAL:
                        source_positions = sent_pe2src[pe_idx]
                        source_sentence_bad_indices |= set(source_positions)
                    elif fluency_rule == FluencyRule.IGNORE_SHF:
                        if pe_sent_tok[pe_idx] not in mt_sent_tok:
                            source_positions = sent_pe2src[pe_idx]
                            source_sentence_bad_indices |= set(source_positions)
                    elif fluency_rule == FluencyRule.MISSING:
                        if mt_idx is None:
                            source_positions = sent_pe2src[pe_idx]
                            source_sentence_bad_indices |= set(source_positions)
                    else:
                        raise Exception(f"Unknown fluency rule {fluency_rule}")
                else:
                    mt_position += 1
            source_sentence_bad_tags = [WMT22QETags.OK.value] * len(src_sent_tok)
            for index in list(source_sentence_bad_indices):
                source_sentence_bad_tags[index] = WMT22QETags.BAD.value
            src_tags.append(source_sentence_bad_tags)

        # Basic sanity checks
        assert all(len(aa) == len(bb) for aa, bb in zip(src_tokens, src_tags)), "SRC tag creation failed"
        return src_tags

    def generate_tags(
        self,
        srcs: List[str],
        mts: List[str],
        pes: List[str],
        src_langs: Union[str, List[str]],
        tgt_langs: Union[str, List[str]],
        use_gaps: bool = False,
        omissions: str = OmissionRule.RIGHT.value,
        fluency_rule: str = FluencyRule.NORMAL.value,
    ) -> Tuple[List[List[TTag]], List[List[TTag]]]:
        src_tokens, src_langs = self.get_tokenized(srcs, src_langs)
        mt_tokens, tgt_langs = self.get_tokenized(mts, tgt_langs)
        pe_tokens, _ = self.get_tokenized(pes, tgt_langs)

        src_pe_alignments = self.align_source_pe(src_tokens, pe_tokens, tgt_langs)
        mt_pe_alignments = self.align_mt_pe(mt_tokens, pe_tokens)

        mt_tags = self.tags_from_edits(mt_tokens, pe_tokens, mt_pe_alignments, use_gaps, omissions)
        src_tags = self.tags_to_source(
            src_tokens,
            pe_tokens,
            mt_tokens,
            src_pe_alignments,
            mt_pe_alignments,
            fluency_rule,
        )

        clear_nlp_cache()

        return src_tags, mt_tags
