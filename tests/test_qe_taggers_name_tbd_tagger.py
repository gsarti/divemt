import sys
from typing import List, Set, Tuple

import pytest

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from divemt.qe_taggers.name_tbd_tagger import NameTBDGeneralTags as Tags
from divemt.qe_taggers.name_tbd_tagger import NameTBDTagger

tagger = NameTBDTagger()


class TestUtils:
    @pytest.mark.parametrize(
        "mt_len, mt_pe_alignments, true_mt_shifts_mask",
        [
            (1, [(0, 0, 0.9)], [False]),
            (2, [(0, 0, 0.9), (1, 1, 0.9)], [False, False]),
            (3, [(0, 0, 0.9), (1, 1, 0.9), (2, 2, 0.9)], [False, False, False]),
            (3, [(0, 0, 0.9), (1, None, None), (2, 1, 0.9)], [False, False, False]),
            # easiest case
            (2, [(0, 1, 0.9), (1, 0, 0.9)], [True, True]),
            # central one is not moved, but have crossing edges
            (3, [(0, 2, 0.9), (1, 1, 0.9), (2, 0, 0.9)], [True, True, True]),
            # the central one deleted, so not shifted, no crossing edges
            (3, [(0, 1, 0.9), (1, None, None), (2, 0, 0.9)], [True, False, True]),
            (
                4,
                [(0, 0, 0.9), (1, 3, 0.9), (1, 4, 0.9), (1, 5, 0.9), (2, 2, 0.9), (2, 0, 0.9), (3, None, None)],
                [False, True, True, False],
            ),
        ],
    )
    def test_detect_crossing_edges(
        self, mt_len: int, mt_pe_alignments: List[Tuple[int, int]], true_mt_shifts_mask: List[bool]
    ) -> None:
        mt_shifts_mask = tagger._detect_crossing_edges(
            [str(i) for i in range(mt_len)], [str(i) for i in range(mt_len)], mt_pe_alignments
        )
        assert mt_shifts_mask == true_mt_shifts_mask


class TestTagsFromEdits:
    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            (
                ["A", "B"],
                ["A", "B"],
                [(0, 0, 0.9), (1, 1, 0.9)],
                [{Tags.OK}, {Tags.OK}],
            ),
            (
                ["A", "B", "C", "D"],
                ["A", "B", "C", "D"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 2, 0.9), (3, 3, 0.9)],
                [{Tags.OK}, {Tags.OK}, {Tags.OK}, {Tags.OK}],
            ),
            ([], [], [], []),
        ],
    )
    def test_single_error_ok(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            (
                ["A", "B", "C"],
                ["A", "X", "Z"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 2, 0.9)],
                [{Tags.OK}, {Tags.BAD_SUBSTITUTION}, {Tags.BAD_SUBSTITUTION}],
            ),
            (
                ["A", "B"],
                ["Z", "X"],
                [(0, 0, 0.9), (1, 1, 0.9)],
                [{Tags.BAD_SUBSTITUTION}, {Tags.BAD_SUBSTITUTION}],
            ),
            # For 1-n and n-1 cases see contraction and expansion tests
        ],
    )
    def test_single_error_substitution(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            (["A", "B"], ["A"], [(0, 0, 0.9), (1, None, None)], [{Tags.OK}, {Tags.BAD_INSERTION}]),
            (["A", "B"], ["B"], [(0, None, None), (1, 0, 0.9)], [{Tags.BAD_INSERTION}, {Tags.OK}]),
            (["A", "B"], [], [(0, None, None), (1, None, None)], [{Tags.BAD_INSERTION}, {Tags.BAD_INSERTION}]),
            # For 1-n and n-1 cases see contraction and expansion tests
        ],
    )
    def test_single_error_insertion(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            (
                ["A"],
                ["A", "X"],
                [(0, 0, 0.9), (None, 1, None)],
                [{Tags.OK, Tags.BAD_DELETION_RIGHT}],
            ),
            (
                ["A"],
                ["X", "A"],
                [(None, 0, None), (0, 1, 0.9)],
                [{Tags.OK, Tags.BAD_DELETION_LEFT}],
            ),
            (
                ["A", "B"],
                ["A", "X", "B"],
                [(0, 0, 0.9), (None, 1, None), (1, 2, 0.9)],
                [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_DELETION_LEFT}],
            ),
            # Delete multiple tokens, but tag error as deleted one
            (
                ["A"],
                ["A", "X", "Y", "Z"],
                [(0, 0, 0.9), (None, 1, None), (None, 2, None), (None, 3, None)],
                [{Tags.OK, Tags.BAD_DELETION_RIGHT}],
            ),
            (
                ["A"],
                ["X", "Y", "Z", "A"],
                [(None, 0, None), (None, 1, None), (None, 2, None), (0, 3, 0.9)],
                [{Tags.OK, Tags.BAD_DELETION_LEFT}],
            ),
            (
                ["A", "B"],
                ["A", "X", "Y", "Z", "B"],
                [(0, 0, 0.9), (None, 1, None), (None, 2, None), (None, 3, None), (1, 4, 0.9)],
                [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_DELETION_LEFT}],
            ),
            # deleted both left and right sides
            (
                ["A"],
                ["X", "A", "Y"],
                [(None, 0, None), (0, 1, 0.9), (None, 2, None)],
                [{Tags.OK, Tags.BAD_DELETION_LEFT, Tags.BAD_DELETION_RIGHT}],
            ),
            # deleted for empty target
            ([], ["X"], [(None, 0, None)], []),
        ],
    )
    def test_single_error_deletion(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            # Have same BBB token, so should filter CCC and TTT out as Deletion error and BBB as Ok
            (
                ["AAA", "BBB"],
                ["AAA", "BBB", "CCC", "TTT"],
                [(0, 0, 0.9), (1, 1, 0.9), (1, 2, 0.1), (1, 3, 0.1)],
                [{Tags.OK}, {Tags.OK, Tags.BAD_DELETION_RIGHT}],
            ),
            (
                ["AAA", "BBB"],
                ["AAA", "TTT", "BBB", "CCC"],
                [(0, 0, 0.9), (1, 1, 0.1), (1, 2, 0.9), (1, 3, 0.1)],
                [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_DELETION_RIGHT, Tags.BAD_DELETION_LEFT}],
            ),
            # XXX, TTT and CCC >threshold are same BBB token, so its bad Contradiction
            (
                ["AAA", "BBB"],
                ["AAA", "XXX", "CCC", "TTT"],
                [(0, 0, 0.9), (1, 1, 0.9), (1, 2, 0.9), (1, 3, 0.9)],
                [{Tags.OK}, {Tags.BAD_CONTRACTION}],
            ),
            # BBX is >threshold, CCC/TTT <threshold, so CCC is Deletion and BBX is Substitution
            (
                ["AAA", "BBB"],
                ["AAA", "BBX", "CCC", "TTT"],
                [(0, 0, 0.9), (1, 1, 0.9), (1, 2, 0.1), (1, 3, 0.1)],
                [{Tags.OK}, {Tags.BAD_SUBSTITUTION, Tags.BAD_DELETION_RIGHT}],
            ),
            (
                ["AAA", "BBB"],
                ["AAA", "TTT", "BBX", "CCC"],
                [(0, 0, 0.9), (1, 1, 0.1), (1, 2, 0.9), (1, 3, 0.1)],
                [
                    {Tags.OK, Tags.BAD_DELETION_RIGHT},
                    {Tags.BAD_SUBSTITUTION, Tags.BAD_DELETION_RIGHT, Tags.BAD_DELETION_LEFT},
                ],
            ),
            # All are >threshold, so all are Contractions
            (
                ["AAA", "BBB"],
                ["AAA", "BBX", "XBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (1, 2, 0.9)],
                [{Tags.OK}, {Tags.BAD_CONTRACTION}],
            ),
            # BBX and XBB >threshold while TTT is <threshold, so its Deletion
            (
                ["AAA", "BBB"],
                ["AAA", "BBX", "TTT", "XBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (1, 2, 0.1), (1, 3, 0.9)],
                [{Tags.OK}, {Tags.BAD_CONTRACTION, Tags.BAD_DELETION_RIGHT}],
            ),
            # TODO: more threshold tests
        ],
    )
    def test_single_error_contraction(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[str]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            # BB token is same, so CCC and TTT are insertions
            (
                ["AAA", "BBB", "CCC", "TTT"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 1, 0.1), (3, 1, 0.1)],
                [{Tags.OK}, {Tags.OK}, {Tags.BAD_INSERTION}, {Tags.BAD_INSERTION}],
            ),
            (
                ["AAA", "TTT", "BBB", "CCC"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.1), (2, 1, 0.9), (3, 1, 0.1)],
                [{Tags.OK}, {Tags.BAD_INSERTION}, {Tags.OK}, {Tags.BAD_INSERTION}],
            ),
            # XXX, TTT and CCC >threshold are same BBB token, so its bad Expansion
            (
                ["AAA", "XXX", "CCC", "TTT"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 1, 0.9), (3, 1, 0.9)],
                [{Tags.OK}, {Tags.BAD_EXPANSION}, {Tags.BAD_EXPANSION}, {Tags.BAD_EXPANSION}],
            ),
            # BBX is >threshold, CCC/TTT <threshold, so CCC is Insertion and BBX is Substitution
            (
                ["AAA", "BBX", "CCC", "TTT"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 1, 0.1), (3, 1, 0.1)],
                [{Tags.OK}, {Tags.BAD_SUBSTITUTION}, {Tags.BAD_INSERTION}, {Tags.BAD_INSERTION}],
            ),
            (
                ["AAA", "CCC", "BBX", "TTT"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.1), (2, 1, 0.9), (3, 1, 0.1)],
                [{Tags.OK}, {Tags.BAD_INSERTION}, {Tags.BAD_SUBSTITUTION}, {Tags.BAD_INSERTION}],
            ),
            # All are >threshold, so all are Expansion
            (
                ["AAA", "BBX", "XBB"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 1, 0.9)],
                [{Tags.OK}, {Tags.BAD_EXPANSION}, {Tags.BAD_EXPANSION}],
            ),
            # BBX and XBB >threshold while TTT is <threshold, so its Insertion
            (
                ["AAA", "BBX", "TTT", "XBB"],
                ["AAA", "BBB"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 1, 0.1), (3, 1, 0.9)],
                [{Tags.OK}, {Tags.BAD_EXPANSION}, {Tags.BAD_INSERTION}, {Tags.BAD_EXPANSION}],
            ),
            # TODO: more threshold tests
        ],
    )
    def test_single_error_expansion(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[str]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags",
        [
            # simple case
            (
                ["A", "B"],
                ["B", "A"],
                [(0, 1, 0.9), (1, 0, 0.9)],
                [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.OK, Tags.BAD_SHIFTING}],
            ),
            # middle intact, but crossing edges, so shifted
            (
                ["A", "X", "Y", "B"],
                ["B", "X", "Y", "A"],
                [(0, 3, 0.9), (1, 1, 0.9), (2, 2, 0.9), (3, 0, 0.9)],
                [
                    {Tags.OK, Tags.BAD_SHIFTING},
                    {Tags.OK, Tags.BAD_SHIFTING},
                    {Tags.OK, Tags.BAD_SHIFTING},
                    {Tags.OK, Tags.BAD_SHIFTING},
                ],
            ),
            # node inserted, so should not be marked as shifted
            (
                ["A", "X", "B"],
                ["B", "A"],
                [(0, 1, 0.9), (1, None, None), (2, 0, 0.9)],
                [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.BAD_INSERTION}, {Tags.OK, Tags.BAD_SHIFTING}],
            ),
            # node deleted, nothing to mark as shifted
            (
                ["A", "B"],
                ["B", "X", "A"],
                [(0, 2, 0.9), (None, 1, None), (1, 0, 0.9)],
                [
                    {Tags.OK, Tags.BAD_SHIFTING, Tags.BAD_DELETION_RIGHT},
                    {Tags.OK, Tags.BAD_SHIFTING, Tags.BAD_DELETION_LEFT},
                ],
            ),
        ],
    )
    def test_single_error_shifted(
        self,
        mt_tokens: List[str],
        pe_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert pred_tags == {t.value for t in true_tags}


class TestTagsToSource:
    @pytest.mark.parametrize(
        "src_tokens, mt_tokens, mt_pe_alignments, mt_tags, true_src_tags",
        [
            # ok cases
            (
                ["A", "B"],
                ["A", "B"],
                [(0, 0, 0.9), (1, 1, 0.9)],
                [{Tags.OK}, {Tags.OK}],
                [{Tags.OK}, {Tags.OK}],
            ),
            (
                ["A", "B", "C", "D"],
                ["A", "B", "C", "D"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 2, 0.9), (3, 3, 0.9)],
                [{Tags.OK}, {Tags.OK}, {Tags.OK}, {Tags.OK}],
                [{Tags.OK}, {Tags.OK}, {Tags.OK}, {Tags.OK}],
            ),
            ([], [], [], [], []),
            # substitution cases
            (
                ["A", "B"],
                ["A", "C"],
                [(0, 0, 0.9), (1, 1, 0.9)],
                [{Tags.OK}, {Tags.BAD_SUBSTITUTION}],
                [{Tags.OK}, {Tags.BAD_SUBSTITUTION}],
            ),
            (
                ["A", "B", "C", "D"],
                ["A", "B", "X", "D"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, 2, 0.1), (3, 3, 0.9)],
                [{Tags.OK}, {Tags.OK}, {Tags.BAD_SUBSTITUTION}, {Tags.OK}],
                [{Tags.OK}, {Tags.OK}, {Tags.BAD_SUBSTITUTION}, {Tags.OK}],
            ),
            # multiple tags
            (
                ["A", "B"],
                ["A", "C"],
                [(0, 0, 0.9), (1, 1, 0.9)],
                [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.BAD_SUBSTITUTION, Tags.BAD_DELETION_RIGHT}],
                [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.BAD_SUBSTITUTION, Tags.BAD_DELETION_RIGHT}],
            ),
        ],
    )
    def test_one_to_one(
        self,
        src_tokens: List[str],
        mt_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        mt_tags: List[Set[StrEnum]],
        true_src_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_to_source(
            [src_tokens], [mt_tokens], [mt_pe_alignments], [[{i.value for i in t} for t in mt_tags]]
        )[0]
        assert len(predicted_tags) == len(true_src_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_src_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "src_tokens, mt_tokens, mt_pe_alignments, mt_tags, true_src_tags",
        [
            (
                ["A"],
                ["A", "B"],
                [(0, 0, 0.9), (None, 1, None)],
                [{Tags.OK}, {Tags.BAD_SUBSTITUTION}],
                [{Tags.OK}],
            ),
            (
                ["A", "B"],
                ["A", "B", "C"],
                [(0, 0, 0.9), (1, 1, 0.9), (None, 2, None)],
                [{Tags.BAD_SUBSTITUTION}, {Tags.OK}, {Tags.OK}],
                [{Tags.BAD_SUBSTITUTION}, {Tags.OK}],
            ),
        ],
    )
    def test_src_deleted(
        self,
        src_tokens: List[str],
        mt_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        mt_tags: List[Set[StrEnum]],
        true_src_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_to_source(
            [src_tokens], [mt_tokens], [mt_pe_alignments], [[{i.value for i in t} for t in mt_tags]]
        )[0]
        assert len(predicted_tags) == len(true_src_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_src_tags):
            assert pred_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize(
        "src_tokens, mt_tokens, mt_pe_alignments, mt_tags, true_src_tags",
        [
            (
                ["A", "B", "C"],
                ["A", "B"],
                [(0, 0, 0.9), (1, 1, 0.9), (2, None, None)],
                [{Tags.BAD_SUBSTITUTION}, {Tags.OK}],
                [{Tags.BAD_SUBSTITUTION}, {Tags.OK}, set()],
            ),
            (
                ["A", "B", "C", "D"],
                ["B"],
                [(0, None, None), (1, 0, 0.9), (2, None, None), (3, None, None)],
                [{Tags.BAD_SUBSTITUTION}],
                [set(), {Tags.BAD_SUBSTITUTION}, set(), set()],
            ),
        ],
    )
    def test_mt_deleted(
        self,
        src_tokens: List[str],
        mt_tokens: List[str],
        mt_pe_alignments: List[Tuple[int, int]],
        mt_tags: List[Set[StrEnum]],
        true_src_tags: List[Set[StrEnum]],
    ) -> None:
        predicted_tags = tagger.tags_to_source(
            [src_tokens], [mt_tokens], [mt_pe_alignments], [[{i.value for i in t} for t in mt_tags]]
        )[0]
        assert len(predicted_tags) == len(true_src_tags)
        for pred_tags, true_tags in zip(predicted_tags, true_src_tags):
            assert pred_tags == {t.value for t in true_tags}
