import codecs
import logging
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import List, Optional, Tuple, Union, Set, Generator
from xml.sax.saxutils import escape

import numpy as np
from simalign import SentenceAligner
from strenum import StrEnum
from tqdm import tqdm
import Levenshtein as lev

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


class NameTBDGeneralTags(StrEnum):
    OK = 'OK'

    BAD_SUBSTITUTION = 'BAD-SUB'
    BAD_DELETION_RIGHT = 'BAD-DEL-R'  # smth deleted on the right side of this token
    BAD_DELETION_LEFT = 'BAD-DEL-L'  # smth deleted on the left side of this token
    BAD_INSERTION = 'BAD-INS'  # 1:n
    BAD_SHIFTING = 'BAD-SHF'  # change words order n:m with hight threshold

    BAD_CONTRACTION = 'BAD-CON'  # 1:n
    BAD_EXPANSION = 'BAD-EXP'


class NameTBDTagger(QETagger):

    ID = "tbd_qe"

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
    ) -> List[List[Tuple[int, int]]]:
        return [
            self.aligner.get_word_aligns(mt_tok, pe_tok)["inter"]
            for mt_tok, pe_tok in tqdm(
                zip(mt_tokens, pe_tokens), total=len(mt_tokens), desc="Aligning mt-pe"
            )
        ]

    @staticmethod
    def _group_by_node(alignments: List[Tuple[Optional[int], Optional[int]]], by_start_node: bool = True, sort: bool = False) -> Generator[Tuple[int, List[int]], None, None]:
        """Yield a node id and a list of connected nodes."""
        _by_index = 0 if by_start_node else 1
        if sort:
            alignments = sorted(alignments, key=lambda x: x[_by_index] if x[_by_index] is not None else -1)
        for start_node, connected_alignments in groupby(alignments, lambda x: x[_by_index]):
            yield start_node, [end_id if by_start_node else start_id for start_id, end_id in connected_alignments]

    @staticmethod
    def _detect_crossing_edges(mt_tokens: List[str], pe_tokens: List[str], alignments: List[Tuple[Optional[int], Optional[int]]]) -> List[bool]:
        """Detect crossing edges in the alignments. Return List of clusters of nodes that are connected."""
        # TODO: optimize from n^2 to n as 2 pointers
        shifted_mt_mask = [False] * len(mt_tokens)

        for i in range(len(alignments)):
            for j in range(i + 1, len(alignments)):
                edge_1, edge_2 = alignments[i], alignments[j]

                # skip if one of the edges is None
                if edge_1[0] is None or edge_1[1] is None or edge_2[0] is None or edge_2[1] is None:
                    continue

                # skip if starting same node
                if edge_1[0] == edge_2[0]:
                    continue

                assert edge_1[0] < edge_2[0], "Alignments have to be are sorted by mt"

                # Check if crossing edges
                if edge_1[0] < edge_2[0] and edge_1[1] > edge_2[1]:
                    # mark the mt token as shifted
                    shifted_mt_mask[edge_1[0]] = True
                    shifted_mt_mask[edge_2[0]] = True

        return shifted_mt_mask

    @staticmethod
    def _lev_similarity(mt_tok: str, pe_tok: str) -> float:
        """Calculate Lev similarity between two tokens in [0, 1] range."""
        if mt_tok == pe_tok:
            return 1.0

        # calculate similarity using Lev distance
        return lev.ratio(mt_tok, pe_tok)

    @staticmethod
    def tags_from_edits(
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        mt_pe_alignments: List[List[Tuple[int, int]]],
        mt_tokens_embeddings: Optional[List[List[np.ndarray]]] = None,
        pe_tokens_embeddings: Optional[List[List[np.ndarray]]] = None,
        threshold: float = 0.5,
    ) -> List[List[Set[str]]]:
        """ Produce tags on MT tokens from edits found in the PE tokens. """
        # TODO: check. now - if embeddings are not provided, use Lev distance
        # TODO: update docs with ERRORS approach rather than EDITS
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
        # - Swapped pair A, B -> B, A: Both blocks receive SHF
        # - For n > 2, all blocks changing relative position receive SHF, others don't

        mt_tags = []
        for mt_tok, pe_tok, mt_pe_align in tqdm(zip(mt_tokens, pe_tokens, mt_pe_alignments), desc="Tagging MT", total=len(mt_tokens)):

            mt_sent_tags: List[Set[str]] = [set() for _ in range(len(mt_tok))]

            # clear 1-n and n-1 nodes with low threshold
            # e.g. if 1-n or n-1 have same token or high similarity, remove low similarity as deletions/insertions
            aligns_remove_1_to_n, aligns_remove_n_to_1 = set(), set()
            # 1-n match
            for mt_node_id, connected_pe_nodes_ids in NameTBDTagger._group_by_node(mt_pe_align, by_start_node=True, sort=False):
                if mt_node_id is not None and len(connected_pe_nodes_ids) > 1:
                    pe_similarity = [
                        (pe_node_id, NameTBDTagger._lev_similarity(mt_tok[mt_node_id], pe_tok[pe_node_id]))
                        for pe_node_id in connected_pe_nodes_ids
                        if pe_node_id is not None
                    ]
                    if all(sim < threshold for _, sim in pe_similarity):
                        continue
                    if all(sim > threshold for _, sim in pe_similarity):
                        continue
                    aligns_remove_1_to_n.update([
                        (mt_node_id, pe_node_id)
                        for pe_node_id, sim in pe_similarity
                        if sim < threshold
                    ])
            # remove selected aligns and add None connected nodes instead
            mt_pe_align = [(None, align[1]) if align in aligns_remove_1_to_n else align for align in mt_pe_align]
            # n-1 match
            for pe_node_id, connected_mt_nodes_ids in NameTBDTagger._group_by_node(mt_pe_align, by_start_node=False, sort=True):
                if pe_node_id is not None and len(connected_mt_nodes_ids) > 1:
                    mt_similarity = [
                        (mt_node_id, NameTBDTagger._lev_similarity(mt_tok[mt_node_id], pe_tok[pe_node_id]))
                        for mt_node_id in connected_mt_nodes_ids
                        if mt_node_id is not None
                    ]
                    if all(sim < threshold for _, sim in mt_similarity):
                        continue
                    if all(sim > threshold for _, sim in mt_similarity):
                        continue
                    aligns_remove_n_to_1.update([
                        (mt_node_id, pe_node_id)
                        for mt_node_id, sim in mt_similarity
                        if sim < threshold
                    ])
            # remove selected aligns and add None connected nodes instead
            mt_pe_align = [(align[0], None) if align in aligns_remove_n_to_1 else align for align in mt_pe_align]

            # Solve all n-1: setup expansions tags and solve n-1 matches < threshold as smth+insertion
            # TODO: check with threshold, now doing without threshold
            for pe_node_id, connected_mt_nodes_ids in NameTBDTagger._group_by_node(mt_pe_align, by_start_node=False, sort=True):
                if pe_node_id is not None and len(connected_mt_nodes_ids) > 1:
                    # expansion, mark related mt nodes
                    for mt_node_id in connected_mt_nodes_ids:
                        if mt_node_id is not None:
                            mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_EXPANSION.value)

            # Solve al deletions, add deletion tags on left and right sides
            mt_position = 0
            for mt_node_id, connected_pe_nodes_ids in NameTBDTagger._group_by_node(mt_pe_align, by_start_node=True, sort=False):
                if mt_node_id is None:
                    # deleted word error, mark left and right modes
                    if 0 <= mt_position - 1 < len(mt_sent_tags):
                        mt_sent_tags[mt_position - 1].add(NameTBDGeneralTags.BAD_DELETION_RIGHT.value)
                    if mt_position < len(mt_sent_tags):
                        mt_sent_tags[mt_position].add(NameTBDGeneralTags.BAD_DELETION_LEFT.value)
                else:
                    mt_position += 1
            # clear all (None, i) to not mess grouping
            mt_pe_align = [align for align in mt_pe_align if align[0] is not None]

            # Solve all 1-n matches
            for mt_node_id, connected_pe_nodes_ids in NameTBDTagger._group_by_node(mt_pe_align, by_start_node=True, sort=True):
                print(mt_node_id, ' -> ', connected_pe_nodes_ids, '\t\tmt_position=', mt_position)
                assert mt_node_id is not None, "Already should be filtered all (None, smth) cases"
                if NameTBDGeneralTags.BAD_EXPANSION.value in mt_sent_tags[mt_node_id]:
                    continue  # TODO: check with gabrielle the priority for EXPANSION and CONTRACTION
                if len(connected_pe_nodes_ids) > 1:
                    # contraction, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_CONTRACTION.value)
                elif connected_pe_nodes_ids[0] is None:
                    # insertion, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_INSERTION.value)
                elif mt_tok[mt_node_id] != pe_tok[connected_pe_nodes_ids[0]]:
                    # substitution, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_SUBSTITUTION.value)
                else:
                    # OK, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.OK.value)

            # Add shifted tags if so
            for mt_node_id, mask in enumerate(NameTBDTagger._detect_crossing_edges(mt_tok, pe_tok, mt_pe_align)):
                if mask:
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_SHIFTING.value)

            # Save tags for this sentence
            mt_tags.append(mt_sent_tags)

        # Basic sanity check
        assert all(
            [len(mt_sent_tokens) == len(mt_sent_tags) for mt_sent_tokens, mt_sent_tags in zip(mt_tokens, mt_tags)]
        ), "MT tags creation failed, number of tokens and tags do not match"
        return mt_tags

    @staticmethod
    def tags_to_source(
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        src_mt_alignments: List[List[Tuple[int, int]]],
        mt_tags: List[List[Set[str]]],
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

    def generate_tags(
        self,
        srcs: List[str],
        mts: List[str],
        pes: List[str],
        src_langs: Union[str, List[Set[str]]],
        tgt_langs: Union[str, List[Set[str]]],
    ) -> Tuple[List[str], List[str]]:
        src_tokens, src_langs = self.get_tokenized(srcs, src_langs)
        mt_tokens, tgt_langs = self.get_tokenized(mts, tgt_langs)
        pe_tokens, _ = self.get_tokenized(pes, tgt_langs)
        src_mt_alignments = self.align_source_mt(src_tokens, mt_tokens, src_langs, tgt_langs)
        mt_pe_alignments = self.align_mt_pe(mt_tokens, pe_tokens, tgt_langs)
        mt_tags = self.tags_from_edits(mt_tokens, pe_tokens, mt_pe_alignments)
        src_tags = self.tags_to_source(
            src_tokens, pe_tokens, src_mt_alignments, mt_tags
        )
        clear_nlp_cache()
        return src_tags, mt_tags
