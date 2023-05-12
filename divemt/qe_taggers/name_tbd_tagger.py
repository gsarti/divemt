import logging
import sys
from itertools import groupby
from typing import Generator, List, Optional, Set, Tuple, Union

import numpy as np

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum
from tqdm import tqdm

from ..cache_utils import CacheDecorator
from .custom_simalign import SentenceAligner as CustomSentenceAligner
from ..parse_utils import clear_nlp_cache
from .base import QETagger, TAlignment, TTag

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _fill_deleted_inserted_tokens(
        len_from_list: List[int], len_to_list: List[int], alignments_list: List[List[TAlignment]]
    ) -> List[List[TAlignment]]:
        """
        As aligner provides only actual alignments, add required i, None), (None, j) tokens
        * (i, None) just inserted in places to maintain order by i
        * (None, j) inserted in estimated places
            - if
        """
        full_new_alignments: List[List[TAlignment]] = []

        for len_from, len_to, alignments in zip(len_from_list, len_to_list, alignments_list):
            new_alignments: List[TAlignment] = []

            # Add (i, None) in correct place (ordered by i)
            current_i_alignment_index = 0
            for align in alignments:
                # Add missing index pairs before current one with (i, None)
                while current_i_alignment_index < align[0]:
                    new_alignments.append((current_i_alignment_index, None, None))
                    current_i_alignment_index += 1

                # Add the current alignment pair
                new_alignments.append(align)
                current_i_alignment_index += 1
            # add last (i, None)
            while current_i_alignment_index < len_from:
                new_alignments.append((current_i_alignment_index, None, None))
                current_i_alignment_index += 1

            # Add (None, j) in correct places
            missed_j_tokens = set(range(len_to)) - {j[1] for j in new_alignments}
            for current_j_alignment_index in missed_j_tokens:
                # select the closest (*, j) by j: obtain index in the list and j value
                closest_value_index = min(
                    range(len(new_alignments)),
                    key=lambda i: abs(new_alignments[i][1] - current_j_alignment_index) if new_alignments[i][1] is not None else np.inf
                )
                closest_value_j = new_alignments[closest_value_index][1]
                # insert position of the (None, current_j_alignment_index) - before of after the closes value
                if closest_value_j < current_j_alignment_index:
                    insert_index = closest_value_index + 1
                else:
                    insert_index = closest_value_index  # - 1
                insert_index = max(0, min(insert_index, len(new_alignments)))
                # insert it in right place
                new_alignments.insert(insert_index, (None, current_j_alignment_index, None))

            full_new_alignments.append(new_alignments)

        return full_new_alignments

    @CacheDecorator()
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

    @CacheDecorator()
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
        alignments: List[TAlignment], by_start_node: bool = True, sort: bool = False
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
        mt_tokens: List[str], pe_tokens: List[str], alignments: List[TAlignment]
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

        mt_tags: List[List[TTag]] = []

        for mt_sent_tok, pe_sent_tok, mt_pe_sent_align in tqdm(
            zip(mt_tokens, pe_tokens, mt_pe_alignments), desc="Tagging MT", total=len(mt_tokens)
        ):
            mt_sent_tags: List[TTag] = [set() for _ in range(len(mt_sent_tok))]

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

        src_tags: List[List[TTag]] = []

        for src_sent_tok, _mt_sent_tok, mt_sent_tags, mt_pe_sent_align in tqdm(
            zip(src_tokens, mt_tokens, mt_tags, src_mt_alignments), desc="Transfer to source", total=len(src_tokens)
        ):
            src_sent_tags: List[TTag] = [set() for _ in range(len(src_sent_tok))]

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
