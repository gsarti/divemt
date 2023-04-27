from typing import List, Tuple, Set

import pytest
from strenum import StrEnum

from divemt.qe_taggers import NameTBDTagger
from divemt.qe_taggers import NameTBDGeneralTags as Tags


class TestUtils:
    @pytest.mark.parametrize("mt_len, mt_pe_alignments, true_mt_shifts_mask", [
        (1, [(0, 0)], [False]),
        (2, [(0, 0), (1, 1)], [False, False]),
        (3, [(0, 0), (1, 1), (2, 2)], [False, False, False]),
        (3, [(0, 0), (1, None), (2, 1)], [False, False, False]),
        # easiest case
        (2, [(0, 1), (1, 0)], [True, True]),
        # central one is not moved, but have crossing edges
        (3, [(0, 2), (1, 1), (2, 0)], [True, True, True]),
        # the central one deleted, so not shifted, no crossing edges
        (3, [(0, 1), (1, None), (2, 0)], [True, False, True]),
        # TODO: check with gabrielle
        (4, [(0, 0), (1, 3), (1, 4), (1, 5), (2, 2), (2, 0), (3, None)], [False, True, True, False]),
    ])
    def test_detect_crossing_edges(self, mt_len: int, mt_pe_alignments: List[Tuple[int, int]], true_mt_shifts_mask: List[bool]) -> None:
        tagger = NameTBDTagger()
        mt_shifts_mask = tagger._detect_crossing_edges([str(i) for i in range(mt_len)], [str(i) for i in range(mt_len)], mt_pe_alignments)
        assert mt_shifts_mask == true_mt_shifts_mask


class TestTagsFromEdits:
    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        (["A", "B"], ["A", "B"], [(0, 0), (1, 1)], [{Tags.OK}, {Tags.OK}]),
        (["A", "B", "C", "D"], ["A", "B", "C", "D"], [(0, 0), (1, 1), (2, 2), (3, 3)], [{Tags.OK}, {Tags.OK}, {Tags.OK}, {Tags.OK}]),
        ([], [], [], []),
    ])
    def test_single_error_ok(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        (["A", "B", "C"], ["A", "X", "Z"], [(0, 0), (1, 1), (2, 2)], [{Tags.OK}, {Tags.BAD_SUBSTITUTION}, {Tags.BAD_SUBSTITUTION}]),
        (["A", "B"], ["Z", "X"], [(0, 0), (1, 1)], [{Tags.BAD_SUBSTITUTION}, {Tags.BAD_SUBSTITUTION}]),
        # For 1-n and n-1 cases see contraction and expansion tests
    ])
    def test_single_error_substitution(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        (["A", "B"], ["A"], [(0, 0), (1, None)], [{Tags.OK}, {Tags.BAD_INSERTION}]),
        (["A", "B"], ["B"], [(0, None), (1, 0)], [{Tags.BAD_INSERTION}, {Tags.OK}]),
        (["A", "B"], [], [(0, None), (1, None)], [{Tags.BAD_INSERTION}, {Tags.BAD_INSERTION}]),
        # For 1-n and n-1 cases see contraction and expansion tests
    ])
    def test_single_error_insertion(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        (["A"], ["A", "X"], [(0, 0), (None, 1)], [{Tags.OK, Tags.BAD_DELETION_RIGHT}]),
        (["A"], ["X", "A"], [(None, 0), (0, 1)], [{Tags.OK, Tags.BAD_DELETION_LEFT}]),
        (["A", "B"], ["A", "X", "B"], [(0, 0), (None, 1), (1, 2)], [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_DELETION_LEFT}]),
        # Delete multiple tokens, but tag error as deleted one
        (["A"], ["A", "X", "Y", "Z"], [(0, 0), (None, 1), (None, 2), (None, 3)], [{Tags.OK, Tags.BAD_DELETION_RIGHT}]),
        (["A"], ["X", "Y", "Z", "A"], [(None, 0), (None, 1), (None, 2), (0, 3)], [{Tags.OK, Tags.BAD_DELETION_LEFT}]),
        (["A", "B"], ["A", "X", "Y", "Z", "B"], [(0, 0), (None, 1), (None, 2), (None, 3), (1, 4)], [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_DELETION_LEFT}]),
        # deleted both left and right sides
        (["A"], ["X", "A", "Y"], [(None, 0), (0, 1), (None, 2)], [{Tags.OK, Tags.BAD_DELETION_LEFT, Tags.BAD_DELETION_RIGHT}]),
        # deleted for empty target
        ([], ["X"], [(None, 0)], []),
    ])
    def test_single_error_deletion(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        # Have same BBB token, so should filter CCC and TTT out as Deletion error and BBB as Ok
        (["AAA", "BBB"], ["AAA", "BBB", "CCC", "TTT"], [(0, 0), (1, 1), (1, 2), (1, 3)], [{Tags.OK}, {Tags.OK, Tags.BAD_DELETION_RIGHT}]),
        (["AAA", "BBB"], ["AAA", "TTT", "BBB", "CCC"], [(0, 0), (1, 1), (1, 2), (1, 3)], [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_DELETION_RIGHT, Tags.BAD_DELETION_LEFT}]),
        # XXX, TTT and CCC >threshold are same BBB token, so its bad Contradiction
        (["AAA", "BBB"], ["AAA", "XXX", "CCC", "TTT"], [(0, 0), (1, 1), (1, 2), (1, 3)], [{Tags.OK}, {Tags.BAD_CONTRACTION}]),
        # BBX is >threshold, CCC/TTT <threshold, so CCC is Deletion and BBX is Substitution
        (["AAA", "BBB"], ["AAA", "BBX", "CCC", "TTT"], [(0, 0), (1, 1), (1, 2), (1, 3)], [{Tags.OK}, {Tags.BAD_SUBSTITUTION, Tags.BAD_DELETION_RIGHT}]),
        (["AAA", "BBB"], ["AAA", "TTT", "BBX", "CCC"], [(0, 0), (1, 1), (1, 2), (1, 3)], [{Tags.OK, Tags.BAD_DELETION_RIGHT}, {Tags.BAD_SUBSTITUTION, Tags.BAD_DELETION_RIGHT, Tags.BAD_DELETION_LEFT}]),
        # All are >threshold, so all are Contractions
        (["AAA", "BBB"], ["AAA", "BBX", "XBB"], [(0, 0), (1, 1), (1, 2)], [{Tags.OK}, {Tags.BAD_CONTRACTION}]),
        # BBX and XBB >threshold while TTT is <threshold, so its Deletion
        (["AAA", "BBB"], ["AAA", "BBX", "TTT", "XBB"], [(0, 0), (1, 1), (1, 2), (1, 3)], [{Tags.OK}, {Tags.BAD_CONTRACTION, Tags.BAD_DELETION_RIGHT}]),
        # TODO: more threshold tests
    ])
    def test_single_error_contraction(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[str]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        # BB token is same, so CCC and TTT are insertions
        (["AAA", "BBB", "CCC", "TTT"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1), (3, 1)], [{Tags.OK}, {Tags.OK}, {Tags.BAD_INSERTION}, {Tags.BAD_INSERTION}]),
        (["AAA", "TTT", "BBB", "CCC"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1), (3, 1)], [{Tags.OK}, {Tags.BAD_INSERTION}, {Tags.OK}, {Tags.BAD_INSERTION}]),
        # XXX, TTT and CCC >threshold are same BBB token, so its bad Expansion
        (["AAA", "XXX", "CCC", "TTT"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1), (3, 1)], [{Tags.OK}, {Tags.BAD_EXPANSION}, {Tags.BAD_EXPANSION}, {Tags.BAD_EXPANSION}]),
        # BBX is >threshold, CCC/TTT <threshold, so CCC is Insertion and BBX is Substitution
        (["AAA", "BBX", "CCC", "TTT"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1), (3, 1)], [{Tags.OK}, {Tags.BAD_SUBSTITUTION}, {Tags.BAD_INSERTION}, {Tags.BAD_INSERTION}]),
        (["AAA", "CCC", "BBX", "TTT"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1), (3, 1)], [{Tags.OK}, {Tags.BAD_INSERTION}, {Tags.BAD_SUBSTITUTION}, {Tags.BAD_INSERTION}]),
        # All are >threshold, so all are Expansion
        (["AAA", "BBX", "XBB"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1)], [{Tags.OK}, {Tags.BAD_EXPANSION}, {Tags.BAD_EXPANSION}]),
        # BBX and XBB >threshold while TTT is <threshold, so its Insertion
        (["AAA", "BBX", "TTT", "XBB"], ["AAA", "BBB"], [(0, 0), (1, 1), (2, 1), (3, 1)], [{Tags.OK}, {Tags.BAD_EXPANSION}, {Tags.BAD_INSERTION}, {Tags.BAD_EXPANSION}]),  # TODO: check priority of Expansion other staff
        # TODO: more threshold tests
    ])
    def test_single_error_expansion(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[str]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}

    @pytest.mark.parametrize("mt_tokens, pe_tokens, mt_pe_alignments, true_mt_tags", [
        # simple case
        (["A", "B"], ["B", "A"], [(0, 1), (1, 0)], [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.OK, Tags.BAD_SHIFTING}]),
        # middle intact, but crossing edges, so shifted
        (["A", "X", "Y", "B"], ["B", "X", "Y", "A"], [(0, 3), (1, 1), (2, 2), (3, 0)], [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.OK, Tags.BAD_SHIFTING}, {Tags.OK, Tags.BAD_SHIFTING}, {Tags.OK, Tags.BAD_SHIFTING}]),
        # node inserted, so should not be marked as shifted TODO: check with gabrielle
        (["A", "X", "B"], ["B", "A"], [(0, 1), (1, None), (2, 0)], [{Tags.OK, Tags.BAD_SHIFTING}, {Tags.BAD_INSERTION}, {Tags.OK, Tags.BAD_SHIFTING}]),
        # node deleted, nothing to mark as shifted
        (["A", "B"], ["B", "X", "A"], [(0, 2), (None, 1), (1, 0)], [{Tags.OK, Tags.BAD_SHIFTING, Tags.BAD_DELETION_RIGHT}, {Tags.OK, Tags.BAD_SHIFTING, Tags.BAD_DELETION_LEFT}]),
    ])
    def test_single_error_shifted(
            self,
            mt_tokens: List[str],
            pe_tokens: List[str],
            mt_pe_alignments: List[Tuple[int, int]],
            true_mt_tags: List[Set[StrEnum]],
    ) -> None:
        tagger = NameTBDTagger()
        predicted_tags = tagger.tags_from_edits([mt_tokens], [pe_tokens], [mt_pe_alignments])[0]
        assert len(predicted_tags) == len(true_mt_tags)
        for predicted_tags, true_tags in zip(predicted_tags, true_mt_tags):
            assert predicted_tags == {t.value for t in true_tags}
