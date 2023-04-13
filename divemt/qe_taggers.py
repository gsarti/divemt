import codecs
import logging
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union
from xml.sax.saxutils import escape

from simalign import SentenceAligner
from strenum import StrEnum
from tqdm import tqdm

from .parse_utils import clear_nlp_cache, tokenize
from .wmt22qe_utils import align_sentence_tercom, parse_tercom_xml_file

logger = logging.getLogger(__name__)


class QETagger(ABC):
    """An abstract class to produce quality estimation tags from src-mt-pe triplets."""

    ID = "qe"

    def align_source_mt(
        self,
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        **align_source_mt_kwargs,
    ) -> List[List[Tuple[int, int]]]:
        """Align source and machine translation tokens."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement align_source_mt()")

    def align_source_pe(
        self,
        src_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        **align_source_pe_kwargs,
    ) -> List[List[Tuple[int, int]]]:
        """Align source and post-edited tokens."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement align_source_pe()")

    @abstractmethod
    def align_mt_pe(
        self,
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        **align_mt_pe_kwargs,
    ) -> List[List[Tuple[int, int]]]:
        """Align machine translation and post-editing tokens."""
        pass

    @staticmethod
    @abstractmethod
    def tags_from_edits(
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        alignments: List[List[Tuple[int, int]]],
        **mt_tagging_kwargs,
    ) -> List[List[str]]:
        """Produce tags on MT tokens from edits found in the PE tokens."""
        pass

    @staticmethod
    @abstractmethod
    def tags_to_source(
        src_tokens: List[List[str]],
        tgt_tokens: List[List[str]],
        **src_tagging_kwargs,
    ) -> List[List[str]]:
        """Propagate tags from MT to source."""
        pass

    @staticmethod
    def get_tokenized(
        sents: List[str], lang: Union[str, List[str]]
    ) -> Tuple[List[List[str]], Union[List[str], List[List[str]]]]:
        """Tokenize sentences."""
        if isinstance(lang, str):
            lang = [lang] * len(sents)
        tok: List[List[str]] = [tokenize(sent, curr_lang, keep_tokens=True) for sent, curr_lang in zip(sents, lang)]
        assert len(tok) == len(lang)
        return tok, lang

    @abstractmethod
    def generate_tags(
        self,
        srcs: List[str],
        mts: List[str],
        pes: List[str],
        src_langs: Union[str, List[str]],
        tgt_langs: Union[str, List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Generate word-level quality estimation tags from source-mt-pe triplets.

        Args:
            srcs (`List[str]`):
                List of untokenized source sentences.
            mts (`List[str]`):
                List of untokenized machine translated sentences.
            pes (`List[str]`):
                List of untokenized post-edited sentences.
            src_langs (`Union[str, List[str]]`):
                Either a single language code for all source sentences or a list of language codes
                (one per source sentence).
            tgt_langs (`Union[str, List[str]]`):
                Either a single language code for all target sentences or a list of language codes
                (one per machine translation).

        Returns:
            `Tuple[List[str], List[str]]`: A tuple containing the lists of quality tags for all source and the machine
            translation sentence, respectively.
        """
        pass


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
    ) -> List[List[Tuple[int, int]]]:
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
    ) -> List[List[Tuple[int, int]]]:
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
        alignments: List[List[Tuple[int, int]]],
        use_gaps: bool = False,
        omissions: str = OmissionRule.RIGHT.value,
    ) -> List[List[str]]:
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
        src_pe_alignments: List[List[Tuple[int, int]]],
        mt_pe_alignments: List[List[Tuple[int, int]]],
        fluency_rule: str = FluencyRule.NORMAL.value,
    ) -> List[List[str]]:
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
    ) -> Tuple[List[List[str]], List[List[str]]]:
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


class NameTBDTagger(QETagger):

    def __init__(
        self,
        aligner: Optional[SentenceAligner] = None,
    ):
        self.aligner = aligner if aligner else SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    def align_source_mt(
        self,
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        src_langs: List[str],
        mt_langs: List[str],
    ) -> List[List[Tuple[int, int]]]:
        return [
            self.aligner.get_word_aligns(src_tok, mt_tok)["inter"]
            for src_tok, mt_tok in tqdm(
                zip(src_tokens, mt_tokens), total=len(src_tokens), desc="Aligning src-mt"
            )
        ]

    def align_mt_pe(
        self,
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        langs: List[str],
    ) -> List[Tuple[int, int]]:
        return [
            self.aligner.get_word_aligns(mt_tok, pe_tok)["inter"]
            for mt_tok, pe_tok in tqdm(
                zip(mt_tokens, pe_tokens), total=len(mt_tokens), desc="Aligning mt-pe"
            )
        ]

    @staticmethod
    def tags_from_edits(
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        alignments: List[List[Tuple[int, int]]],
    ) -> List[List[str]]:
        """ Produce tags on MT tokens from edits found in the PE tokens. """
        # 1:1 match: OK if same, SUB if different
        # 1:n match:
        # - Find highest match for 1 in n (lexical, LaBSE if not found)
        # - If all matches are < threshold, tag as EXP (expansion)
        # - Else, assign OK if same, SUB if different
        # - If match preceded by some of the n, assign also INS to match
        # - If match followed by some of the n, push an INS tag to the next token
        # n:1 match:
        # - Find highest match for 1 in n (lexical, LaBSE if not found)
        # - If all matches are < threshold, tag as CON (contraction)
        # - Else, assign OK if same, SUB if different
        # - All n different than match are assigned DEL
        # n:m match:
        # - For each 1 in n, find highest match for 1 in m (lexical, LaBSE if not found, from highest score to lowest)
        # - If all matches are < threshold, skip and continue
        # - Else assign OK if same, SUB if different, remove from available m matches
        # If in a block with multiple crossing alignments (with blocks named A, B, ...):
        # - Swapped pair A, B -> B, A: Both blocks recive SHF
        # - For n > 2, all blocks changing relative position recive SHF, others don't
        raise NotImplementedError()

    @staticmethod
    def tags_to_source(
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        alignments: List[List[Tuple[int, int]]],
        mt_tags: List[List[str]],
    ) -> List[List[str]]:
        """ Propagate tags from MT to source. """
        # 1:1 match: copy tags from MT
        # 1:n match:
        # - Find highest match for 1 in n (lexical, LaBSE if not found)
        # - If all matches are < threshold, TBD
        # - Else, copy tags from top match in MT and ignore other matches
        # n:1 match: copy tags from 1 to all n
        # n:m match:
        # - For each 1 in n, find highest match for 1 in m (lexical, LaBSE if not found)
        # - If all matches are < threshold, ignore and continue
        # - Copy tags from top match in MT and ignore other matches
        raise NotImplementedError()
