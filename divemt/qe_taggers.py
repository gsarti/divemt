import codecs
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Generator, List, Optional, Set, Tuple, Union
from xml.sax.saxutils import escape

from simalign import SentenceAligner

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum
from tqdm import tqdm

from .custom_simalign import SentenceAligner as CustomSentenceAligner
from .parse_utils import clear_nlp_cache, tokenize
from .wmt22qe_utils import align_sentence_tercom, parse_tercom_xml_file

logger = logging.getLogger(__name__)


TTag = Union[str, Set[str]]
TAlignment = Union[Tuple[Optional[int], Optional[int]], Tuple[Optional[int], Optional[int], Optional[float]]]


class QETagger(ABC):
    """An abstract class to produce quality estimation tags from src-mt-pe triplets."""

    ID = "qe"

    def align_source_mt(
        self,
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        **align_source_mt_kwargs: Any,
    ) -> List[List[TAlignment]]:
        """Align source and machine translation tokens."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement align_source_mt()")

    def align_source_pe(
        self,
        src_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        **align_source_pe_kwargs: Any,
    ) -> List[List[TAlignment]]:
        """Align source and post-edited tokens."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement align_source_pe()")

    @abstractmethod
    def align_mt_pe(
        self,
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        **align_mt_pe_kwargs: Any,
    ) -> List[List[TAlignment]]:
        """Align machine translation and post-editing tokens."""
        pass

    @staticmethod
    @abstractmethod
    def tags_from_edits(
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        alignments: List[List[TAlignment]],
        **mt_tagging_kwargs: Any,
    ) -> List[List[TTag]]:
        """Produce tags on MT tokens from edits found in the PE tokens."""
        pass

    @staticmethod
    @abstractmethod
    def tags_to_source(
        src_tokens: List[List[str]],
        tgt_tokens: List[List[str]],
        **src_tagging_kwargs: Any,
    ) -> List[List[TTag]]:
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
    ) -> Tuple[List[TTag], List[TTag]]:
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
            `Tuple[List[TTag], List[TTag]]`: A tuple containing the lists of quality tags for all source and the
            machine translation sentence, respectively.
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


class NameTBDGeneralTags(StrEnum):
    """Error types tags for NameTBD."""

    OK = "OK"  # 1:1 - the MT uses the same single word as the PE
    BAD_SUBSTITUTION = "BAD-SUB"  # 1:1 - the MT uses a different single word than the PE

    BAD_DELETION_RIGHT = "BAD-DEL-R"  # None:1 - the MT does not have a word existed in PE, deletion on the right
    BAD_DELETION_LEFT = "BAD-DEL-L"  # None:1 - the MT does not have a word existed in PE, deletion on the left
    BAD_INSERTION = "BAD-INS"  # 1:None - the MT wrongly inserted a words that is not in the PE

    BAD_SHIFTING = "BAD-SHF"  # for any number of tokens - detect crossing edges

    BAD_CONTRACTION = "BAD-CON"  # 1:n - the MT uses a single word instead of multiple words in the PE
    BAD_EXPANSION = "BAD-EXP"  # n:1 - the MT uses a multiple words instead of one in the PE


class NameTBDTagger(QETagger):
    ID = "tbd_qe"

    def __init__(
        self,
        aligner: Optional[CustomSentenceAligner] = None,
    ):
        self.aligner = (
            aligner
            if aligner
            else CustomSentenceAligner(model="bert", token_type="bpe", matching_methods="mai", return_similarity="avg")
        )

    def _fill_deleted_inserted_tokens(
        self, len_from: int, len_to: int, alignments: List[TAlignment]
    ) -> List[TAlignment]:
        """As aligner provides only actual alignments, add required (None, i), (i, None) tokens"""
        new_alignments: List[TAlignment] = []

        # Add (i, None) in correct place (ordered by i)
        current_alignment_index = 0
        for align in alignments:
            # Add missing index pairs with None
            while current_alignment_index < align[0]:
                new_alignments.append((current_alignment_index, None))
                current_alignment_index += 1

            # Add the current alignment pair
            new_alignments.append(align)
            current_alignment_index += 1

        raise NotImplementedError()

        return new_alignments

    # @CacheDecorator()
    def align_source_mt(
        self,
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        src_langs: List[str],
        mt_langs: List[str],
    ) -> List[List[TAlignment]]:
        return [
            self.aligner.get_word_aligns(src_tok, mt_tok)["inter"]
            for src_tok, mt_tok in tqdm(zip(src_tokens, mt_tokens), total=len(src_tokens), desc="Aligning src-mt")
        ]

    # @CacheDecorator()
    def align_mt_pe(
        self,
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        langs: List[str],
    ) -> List[List[TAlignment]]:
        return [
            self.aligner.get_word_aligns(mt_tok, pe_tok)["inter"]
            for mt_tok, pe_tok in tqdm(zip(mt_tokens, pe_tokens), total=len(mt_tokens), desc="Aligning mt-pe")
        ]

    @staticmethod
    def _group_by_node(
        alignments: List[Tuple[Optional[int], Optional[int]]], by_start_node: bool = True, sort: bool = False
    ) -> Generator[Tuple[int, List[int], List[float]], None, None]:
        """Yield a node id and a list of connected nodes."""
        _by_index = 0 if by_start_node else 1
        if sort:
            alignments = sorted(alignments, key=lambda x: x[_by_index] if x[_by_index] is not None else -1)
        for start_node, connected_alignments in groupby(alignments, lambda x: x[_by_index]):
            connected_alignments = list(connected_alignments)
            yield start_node, [
                end_id if by_start_node else start_id for start_id, end_id, _ in connected_alignments
            ], [similarity for _, _, similarity in connected_alignments]

    @staticmethod
    def _detect_crossing_edges(
        mt_tokens: List[str], pe_tokens: List[str], alignments: List[Tuple[Optional[int], Optional[int], float]]
    ) -> List[bool]:
        """Detect crossing edges in the alignments. Return mask list of nodes that cross some other node."""
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
    def tags_from_edits(
        mt_tokens: List[List[str]],
        pe_tokens: List[List[str]],
        mt_pe_alignments: List[List[TAlignment]],
        threshold: float = 0.8,
    ) -> List[List[TTag]]:
        """Produce tags on MT tokens from edits found in the PE tokens.

        Note: The tags indicate the type of error particular MT token is affected by.

        The following situations are considered:
            1:1 match: OK if same, SUB if different
            1:n match:
            - Obtain similarity between 1 and n (lexical, LaBSE if not found)
            - If all matches are <threshold or all >threshold, tag as CON (contraction)
            - Else, tackle the highest match as 1:1 (OK/SUB) and the rest as None:1 (deletions)
            n:1 match:
            - Obtain similarity between n and 1 (lexical, LaBSE if not found)
            - If all matches are <threshold or all >threshold, tag as EXP (expansion)
            - Else, tackle the highest match as 1:1 (OK/SUB) and the rest as 1:None (insertions)
            n:m match:
            - Prioritize n:1 matches with the EXP (expansion) tag
            - Clear all None:1 cases
            - Consider all n:m as 1:m cases, if current MT token is not tagged as EXP
            shifting:
            - First, clear all None:1 and 1:None cases - deleted and inserted words can't be shifted
            - Then for all edges check if they cross with any other edge
            - If they do, mark both nodes (2 edges starting node) in MT as SHF (shifted)
            - TODO:
                If in a block with multiple crossing alignments (with blocks named A, B, ...):
                - Swapped pair A, B -> B, A: Both blocks receive SHF
                - For n > 2, all blocks changing relative position receive SHF, others don't
        """
        # TODO: check. now - if embeddings are not provided, use Lev distance

        mt_tags: List[List[Set[str]]] = []

        for mt_sent_tok, pe_sent_tok, mt_pe_sent_align in tqdm(
            zip(mt_tokens, pe_tokens, mt_pe_alignments), desc="Tagging MT", total=len(mt_tokens)
        ):
            mt_sent_tags: List[Set[str]] = [set() for _ in range(len(mt_sent_tok))]

            # clear 1-n and n-1 nodes with low threshold
            # e.g. if 1-n or n-1 have same token or high similarity, remove low similarity as deletions/insertions
            # (None:1 and 1:None)
            aligns_remove_1_to_n, aligns_remove_n_to_1 = set(), set()
            # 1-n match
            for mt_node_id, connected_pe_nodes_ids, connected_pe_similarity in NameTBDTagger._group_by_node(
                mt_pe_sent_align, by_start_node=True, sort=True
            ):
                if mt_node_id is not None and len(connected_pe_nodes_ids) > 1:
                    if all(sim < threshold for sim in connected_pe_similarity if sim is not None):
                        continue
                    if all(sim > threshold for sim in connected_pe_similarity if sim is not None):
                        continue
                    aligns_remove_1_to_n.update(
                        [
                            (mt_node_id, pe_node_id, sim)
                            for pe_node_id, sim in zip(connected_pe_nodes_ids, connected_pe_similarity)
                            if pe_node_id is not None and sim is not None and sim < threshold
                        ]
                    )
            # remove selected aligns and add None connected nodes instead
            mt_pe_sent_align = [
                (None, align[1], None) if align in aligns_remove_1_to_n else align for align in mt_pe_sent_align
            ]
            # n-1 match
            for pe_node_id, connected_mt_nodes_ids, connected_mt_similarity in NameTBDTagger._group_by_node(
                mt_pe_sent_align, by_start_node=False, sort=True
            ):
                if pe_node_id is not None and len(connected_mt_nodes_ids) > 1:
                    if all(sim < threshold for sim in connected_mt_similarity if sim is not None):
                        continue
                    if all(sim > threshold for sim in connected_mt_similarity if sim is not None):
                        continue
                    aligns_remove_n_to_1.update(
                        [
                            (mt_node_id, pe_node_id, sim)
                            for mt_node_id, sim in zip(connected_mt_nodes_ids, connected_mt_similarity)
                            if mt_node_id is not None and sim is not None and sim < threshold
                        ]
                    )
            # remove selected aligns and add None connected nodes instead
            mt_pe_sent_align = [
                (align[0], None, None) if align in aligns_remove_n_to_1 else align for align in mt_pe_sent_align
            ]

            # Solve all n-1: setup expansions tags and solve n-1 matches < threshold as smth+insertion
            for pe_node_id, connected_mt_nodes_ids, _ in NameTBDTagger._group_by_node(
                mt_pe_sent_align, by_start_node=False, sort=True
            ):
                if pe_node_id is not None and len(connected_mt_nodes_ids) > 1:
                    # expansion, mark related mt nodes
                    for mt_node_id in connected_mt_nodes_ids:
                        if mt_node_id is not None:
                            mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_EXPANSION.value)

            # Solve all deletions, add deletion tags on left and right sides
            mt_position = 0
            for mt_node_id, connected_pe_nodes_ids, _ in NameTBDTagger._group_by_node(
                mt_pe_sent_align, by_start_node=True, sort=False
            ):
                if mt_node_id is None:
                    # deleted word error, mark left and right modes
                    if 0 <= mt_position - 1 < len(mt_sent_tags):
                        mt_sent_tags[mt_position - 1].add(NameTBDGeneralTags.BAD_DELETION_RIGHT.value)
                    if mt_position < len(mt_sent_tags):
                        mt_sent_tags[mt_position].add(NameTBDGeneralTags.BAD_DELETION_LEFT.value)
                else:
                    mt_position += 1
            # clear all (None, i) to not mess grouping
            mt_pe_sent_align = [align for align in mt_pe_sent_align if align[0] is not None]

            # Solve all 1-n matches
            for mt_node_id, connected_pe_nodes_ids, _ in NameTBDTagger._group_by_node(
                mt_pe_sent_align, by_start_node=True, sort=True
            ):
                assert mt_node_id is not None, "Already should be filtered all (None, smth) cases"
                if NameTBDGeneralTags.BAD_EXPANSION.value in mt_sent_tags[mt_node_id]:
                    continue
                if len(connected_pe_nodes_ids) > 1:
                    # contraction, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_CONTRACTION.value)
                elif connected_pe_nodes_ids[0] is None:
                    # insertion, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_INSERTION.value)
                elif mt_sent_tok[mt_node_id] != pe_sent_tok[connected_pe_nodes_ids[0]]:
                    # substitution, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_SUBSTITUTION.value)
                else:
                    # OK, mark the node
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.OK.value)

            # Add shifted tags if so
            for mt_node_id, mask in enumerate(
                NameTBDTagger._detect_crossing_edges(mt_sent_tok, pe_sent_tok, mt_pe_sent_align)
            ):
                if mask:
                    mt_sent_tags[mt_node_id].add(NameTBDGeneralTags.BAD_SHIFTING.value)

            # Save tags for this sentence
            mt_tags.append(mt_sent_tags)

        # Basic sanity check
        assert all(
            len(mt_sent_tokens) == len(mt_sent_tags) for mt_sent_tokens, mt_sent_tags in zip(mt_tokens, mt_tags)
        ), "MT tags creation failed, number of tokens and tags do not match"
        return mt_tags

    @staticmethod
    def tags_to_source(
        src_tokens: List[List[str]],
        mt_tokens: List[List[str]],
        src_mt_alignments: List[List[TAlignment]],
        mt_tags: List[List[Set[str]]],
    ) -> List[List[TTag]]:
        """Propagate tags from MT to source.

        # TODO: update docstring with the final logic
        The following cases are considered:
            1:1 match: copy tags from MT
            1:n match:
            - Find highest match for 1 in n (lexical, LaBSE if not found)
            - If all matches are <threshold or all >threshold, TBD
            - Else, copy tags from top match in MT and ignore other matches
            n:1 match: copy tags from 1 to all n
            n:m match:
            - For each 1 in n, find highest match for 1 in m (lexical, LaBSE if not found)
            - If all matches are <threshold or all >threshold, ignore and continue
            - Copy tags from top match in MT and ignore other matches
        """

        src_tags: List[List[Set[str]]] = []

        for src_sent_tok, _mt_sent_tok, mt_sent_tags, mt_pe_sent_align in tqdm(
            zip(src_tokens, mt_tokens, mt_tags, src_mt_alignments), desc="Transfer to source", total=len(src_tokens)
        ):
            src_sent_tags: List[Set[str]] = [set() for _ in range(len(src_sent_tok))]

            # Solve all as 1-n matches
            for src_node_id, connected_mt_nodes_ids, connected_mt_similarity in NameTBDTagger._group_by_node(
                mt_pe_sent_align, by_start_node=True, sort=True
            ):
                if src_node_id is None:
                    continue
                elif len(connected_mt_nodes_ids) == 0:
                    continue
                elif len(connected_mt_nodes_ids) > 1:
                    # n-1 match, find best match
                    best_mt_node_id, best_mt_similarity = None, 0.0
                    for mt_node_id, mt_similarity in zip(connected_mt_nodes_ids, connected_mt_similarity):
                        if mt_similarity is not None and mt_similarity > best_mt_similarity:
                            best_mt_node_id, best_mt_similarity = mt_node_id, mt_similarity
                    if best_mt_node_id is None:
                        # no good match, ignore
                        continue
                    else:
                        # copy tags from best match
                        src_sent_tags[src_node_id].update(mt_sent_tags[best_mt_node_id])
                elif connected_mt_nodes_ids[0] is None:
                    # nothing to copy from MT
                    continue
                else:
                    # 1-1 match, copy tags
                    src_sent_tags[src_node_id].update(mt_sent_tags[connected_mt_nodes_ids[0]])

            # Save tags for this sentence
            src_tags.append(src_sent_tags)

        # Basic sanity checks
        assert all(
            len(aa) == len(bb) for aa, bb in zip(src_tokens, src_tags)
        ), "Source tags creation failed, number of tokens and tags do not match"
        return src_tags

    def generate_tags(
        self,
        srcs: List[str],
        mts: List[str],
        pes: List[str],
        src_langs: Union[str, List[Set[str]]],
        tgt_langs: Union[str, List[Set[str]]],
    ) -> Tuple[List[TTag], List[TTag]]:
        src_tokens, src_langs = self.get_tokenized(srcs, src_langs)
        mt_tokens, tgt_langs = self.get_tokenized(mts, tgt_langs)
        pe_tokens, _ = self.get_tokenized(pes, tgt_langs)

        src_mt_alignments = self.align_source_mt(src_tokens, mt_tokens, src_langs, tgt_langs)
        mt_pe_alignments = self.align_mt_pe(mt_tokens, pe_tokens, tgt_langs)

        mt_tags = self.tags_from_edits(mt_tokens, pe_tokens, mt_pe_alignments)
        src_tags = self.tags_to_source(src_tokens, pe_tokens, src_mt_alignments, mt_tags)

        clear_nlp_cache()

        return src_tags, mt_tags
