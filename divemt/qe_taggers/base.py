from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set, Tuple, Union

from ..parse_utils import tokenize

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
