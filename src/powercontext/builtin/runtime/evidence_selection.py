# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select literal query-relevant sentence spans from candidate Memory texts.

The selector is pure and local: it reads only the request query and the candidate texts the
retrieval stage already provided, keeps exact Python character offsets into those texts, and
never rewrites, translates, or summarizes a single character. Ranking is a small BM25 field over
sentence spans so rare query terms dominate common ones; packing calls a caller-supplied
``measure`` callback for the exact rendered byte cost, because the caller owns the trust policy,
envelope, and citation format that the budget must cover.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_TERMINATORS = frozenset({".", "!", "?"})
_CLOSERS = frozenset({'"', "'", ")", "]", "}"})
_ABBREVIATIONS = frozenset({
    "al",
    "am",
    "approx",
    "avg",
    "ca",
    "cf",
    "co",
    "corp",
    "dept",
    "div",
    "dr",
    "e.g",
    "e.u",
    "eds",
    "eq",
    "eqs",
    "est",
    "etc",
    "fig",
    "figs",
    "i.e",
    "inc",
    "jr",
    "ltd",
    "max",
    "min",
    "mr",
    "mrs",
    "ms",
    "no",
    "nos",
    "pm",
    "pp",
    "prof",
    "ref",
    "refs",
    "resp",
    "sec",
    "seq",
    "sr",
    "spp",
    "st",
    "subsp",
    "tab",
    "u.k",
    "u.n",
    "u.s",
    "univ",
    "var",
    "vol",
    "vs",
})

# BM25 term saturation and length normalization.
_K1 = 1.2
_B = 0.75
# A title/lead sentence is rarely itself cited evidence, and fragment-like captions are noisy.
_FIRST_SENTENCE_WEIGHT = 0.8
_SHORT_SENTENCE_TOKENS = 3
_SHORT_SENTENCE_WEIGHT = 0.75
# Adjacent query words appearing adjacently in a span are much stronger evidence than a bag match.
_PHRASE_BONUS_WEIGHT = 0.6
# A sentence matching more of the query's weighted terms is likelier to be the evidence sentence
# even when every single term is common in the candidate set (a paper repeats its own topic words).
_COVERAGE_WEIGHT = 0.6
_MAX_SPANS = 8


@dataclass(frozen=True)
class EvidenceSpan:
    """One literal half-open character span of one candidate text."""

    candidate_index: int
    start: int
    end: int


@dataclass(frozen=True)
class _Sentence:
    start: int
    end: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class _Unit:
    candidate_index: int
    sentence_index: int
    score: float


def split_sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Split ``text`` into trimmed sentence spans, keeping exact Python character offsets.

    Splitting keeps the terminating punctuation and any closing quotes/brackets in the span, so a
    sentence span is a superset of the sentence an annotation would mark. Abbreviations, single
    letter initials, decimals, and lowercase continuations never end a sentence.
    """

    spans: list[tuple[int, int]] = []
    length = len(text)
    start = 0
    cursor = 0
    while cursor < length:
        if text[cursor] not in _TERMINATORS:
            cursor += 1
            continue
        terminator = cursor
        cursor += 1
        while cursor < length and text[cursor] in _CLOSERS:
            cursor += 1
        if cursor >= length:
            _append_span(spans, text, start, cursor)
            start = cursor
            break
        if not text[cursor].isspace():
            continue
        next_start = cursor
        while next_start < length and text[next_start].isspace():
            next_start += 1
        if _sentence_continues(text, terminator, next_start):
            continue
        _append_span(spans, text, start, cursor)
        start = next_start
    _append_span(spans, text, start, length)
    return tuple(spans)


def select_evidence_spans(
    query: str,
    texts: Sequence[str],
    *,
    max_bytes: int,
    measure: Callable[[Sequence[EvidenceSpan]], int],
    max_spans: int = _MAX_SPANS,
) -> tuple[EvidenceSpan, ...]:
    """Select literal sentence spans that fit ``max_bytes`` and answer ``query``.

    ``measure`` must return the exact rendered UTF-8 byte count the caller would produce for the
    spans, including its trust policy, envelope, and citations. The returned spans are disjoint
    per candidate, non-decreasing in candidate order, and every one is a literal slice of its
    candidate text.
    """

    sentences = tuple(_sentences(text) for text in texts)
    units = _rank_units(query, sentences)
    chosen: set[tuple[int, int]] = set()

    def spans_for(chosen_sentences: set[tuple[int, int]]) -> tuple[EvidenceSpan, ...]:
        return _merge_spans(chosen_sentences, sentences)

    def try_add(candidate_index: int, sentence_index: int) -> bool:
        if (candidate_index, sentence_index) in chosen:
            return False
        trial = chosen | {(candidate_index, sentence_index)}
        trial_spans = spans_for(trial)
        if len(trial_spans) > max_spans:
            return False
        if measure(trial_spans) > max_bytes:
            return False
        chosen.clear()
        chosen.update(trial)
        return True

    # Seed with every sentence the query actually matches, best first: a matched sentence is a
    # far better candidate than any unmatched one, whether it starts a new fragment or extends one.
    for unit in units:
        if unit.score > 0.0:
            try_add(unit.candidate_index, unit.sentence_index)

    _fill_spare_budget(chosen, units, sentences, try_add)

    return spans_for(chosen)


def _sentences(text: str) -> tuple[_Sentence, ...]:
    return tuple(
        _Sentence(start=start, end=end, tokens=_tokens(text[start:end])) for start, end in split_sentence_spans(text)
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text))


def _token_variants(token: str) -> frozenset[str]:
    """Return cheap, conservative stem variants so inflected query and evidence words still meet."""

    variants = {token}
    if len(token) <= 3:
        return frozenset(variants)
    if token.endswith("ies") and len(token) > 4:
        variants.add(f"{token[:-3]}y")
    if token.endswith("es"):
        variants.add(token[:-2])
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        variants.add(token[:-1])
    if token.endswith("ing") and len(token) > 5:
        variants.add(token[:-3])
        variants.add(f"{token[:-3]}e")
    if token.endswith("ed") and len(token) > 4:
        variants.add(token[:-2])
        variants.add(f"{token[:-2]}e")
    if token.endswith("ation") and len(token) > 7:
        variants.add(f"{token[:-5]}e")
        variants.add(token[:-5])
    if token.endswith("ations") and len(token) > 8:
        variants.add(f"{token[:-6]}e")
        variants.add(token[:-6])
    return frozenset(variants)


def _rank_units(query: str, sentences: Sequence[Sequence[_Sentence]]) -> tuple[_Unit, ...]:
    units: list[_Unit] = []
    unit_tokens: list[tuple[str, ...]] = []
    for candidate_index, candidate_sentences in enumerate(sentences):
        for sentence_index, sentence in enumerate(candidate_sentences):
            units.append(_Unit(candidate_index=candidate_index, sentence_index=sentence_index, score=0.0))
            unit_tokens.append(sentence.tokens)
    query_tokens = _tokens(query)
    if not units or not query_tokens:
        return tuple(units)

    terms: dict[str, frozenset[str]] = {}
    for token in query_tokens:
        terms.setdefault(token, _token_variants(token))
    ordered_terms = tuple(terms)
    counts_per_unit: list[dict[str, int]] = []
    for tokens in unit_tokens:
        counts_per_unit.append(_term_counts(tokens, terms))
    document_frequency = {
        term: sum(1 for counts in counts_per_unit if counts.get(term, 0) > 0) for term in ordered_terms
    }
    unit_count = len(units)
    inverse_frequency = {
        term: math.log(1.0 + (unit_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
        if frequency > 0
    }
    average_length = sum(len(tokens) for tokens in unit_tokens) / unit_count or 1.0
    phrases = _phrase_terms(query_tokens, inverse_frequency)
    total_weight = sum(inverse_frequency.values())

    ranked: list[_Unit] = []
    for unit, tokens, counts in zip(units, unit_tokens, counts_per_unit, strict=True):
        length = len(tokens)
        score = 0.0
        matched_weight = 0.0
        for term, count in counts.items():
            weight = inverse_frequency.get(term)
            if weight is None or count == 0:
                continue
            matched_weight += weight
            score += weight * count * (_K1 + 1.0) / (count + _K1 * (1.0 - _B + _B * length / average_length))
        for first, second, weight in phrases:
            if _contains_phrase(tokens, terms[first], terms[second]):
                score += weight
        if total_weight > 0.0:
            score *= 1.0 + _COVERAGE_WEIGHT * matched_weight / total_weight
        if unit.sentence_index == 0:
            score *= _FIRST_SENTENCE_WEIGHT
        if length <= _SHORT_SENTENCE_TOKENS:
            score *= _SHORT_SENTENCE_WEIGHT
        ranked.append(_Unit(unit.candidate_index, unit.sentence_index, score))
    ranked.sort(key=lambda unit: (-unit.score, unit.candidate_index, unit.sentence_index))
    return tuple(ranked)


def _term_counts(tokens: Sequence[str], terms: dict[str, frozenset[str]]) -> dict[str, int]:
    counts = dict.fromkeys(terms, 0)
    for token in tokens:
        variants = _token_variants(token)
        if len(variants) == 1:
            only = next(iter(variants))
            if only in counts:
                counts[only] += 1
            continue
        for term, term_variants in terms.items():
            if variants & term_variants:
                counts[term] += 1
    return counts


def _phrase_terms(
    query_tokens: Sequence[str], inverse_frequency: dict[str, float]
) -> tuple[tuple[str, str, float], ...]:
    phrases: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for first, second in zip(query_tokens, query_tokens[1:], strict=False):
        if first == second or (first, second) in seen:
            continue
        first_weight = inverse_frequency.get(first)
        second_weight = inverse_frequency.get(second)
        if first_weight is None or second_weight is None:
            continue
        seen.add((first, second))
        phrases.append((first, second, _PHRASE_BONUS_WEIGHT * 0.5 * (first_weight + second_weight)))
    return tuple(phrases)


def _contains_phrase(tokens: Sequence[str], first: frozenset[str], second: frozenset[str]) -> bool:
    for left, right in zip(tokens, tokens[1:], strict=False):
        if _token_variants(left) & first and _token_variants(right) & second:
            return True
    return False


def _merge_spans(
    chosen: set[tuple[int, int]],
    sentences: Sequence[Sequence[_Sentence]],
) -> tuple[EvidenceSpan, ...]:
    spans: list[EvidenceSpan] = []
    for candidate_index, candidate_sentences in enumerate(sentences):
        selected = sorted(
            sentence_index
            for chosen_candidate, sentence_index in chosen
            if chosen_candidate == candidate_index
        )
        block: list[int] = []
        for sentence_index in selected:
            if block and sentence_index != block[-1] + 1:
                spans.append(
                    EvidenceSpan(
                        candidate_index=candidate_index,
                        start=candidate_sentences[block[0]].start,
                        end=candidate_sentences[block[-1]].end,
                    )
                )
                block = []
            block.append(sentence_index)
        if block:
            spans.append(
                EvidenceSpan(
                    candidate_index=candidate_index,
                    start=candidate_sentences[block[0]].start,
                    end=candidate_sentences[block[-1]].end,
                )
            )
    return tuple(spans)


def _fill_spare_budget(
    chosen: set[tuple[int, int]],
    units: Sequence[_Unit],
    sentences: Sequence[Sequence[_Sentence]],
    try_add: Callable[[int, int], bool],
) -> None:
    """Spend leftover bytes on sentences adjacent to an already selected one.

    Unused budget is wasted budget, and the sentence next to a selected sentence is the most
    likely place for the rest of the same evidence block. Unmatched sentences far from every
    selection are not worth the citation overhead they would add.
    """

    if not chosen:
        for candidate_index, candidate_sentences in enumerate(sentences):
            if not candidate_sentences:
                continue
            if try_add(candidate_index, 0):
                break
    scores = {(unit.candidate_index, unit.sentence_index): unit.score for unit in units}
    while True:
        neighbors: list[tuple[float, int, int, int]] = []
        for candidate_index, sentence_index in chosen:
            for direction, adjacent in ((1, sentence_index + 1), (-1, sentence_index - 1)):
                if adjacent < 0 or adjacent >= len(sentences[candidate_index]):
                    continue
                if (candidate_index, adjacent) in chosen:
                    continue
                # Prefer continuing forward: evidence blocks usually continue after the sentence
                # that matched the query most strongly.
                forward_rank = 0 if direction == 1 else 1
                neighbors.append(
                    (scores.get((candidate_index, adjacent), 0.0), candidate_index, forward_rank, adjacent)
                )
        if not neighbors:
            return
        neighbors.sort(key=lambda entry: (-entry[0], entry[1], entry[2], entry[3]))
        progressed = False
        for _, candidate_index, _, adjacent in neighbors:
            if try_add(candidate_index, adjacent):
                progressed = True
                break
        if not progressed:
            return


def _append_span(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))


def _sentence_continues(text: str, terminator: int, next_start: int) -> bool:
    if next_start < len(text) and text[next_start].islower():
        return True
    word_end = terminator
    word_start = word_end - 1
    while word_start >= 0 and text[word_start].isspace():
        word_start -= 1
    word_end = word_start + 1
    while word_start >= 0 and (text[word_start].isalnum() or text[word_start] in ".-"):
        word_start -= 1
    word = text[word_start + 1 : word_end].strip(".")
    if not word:
        return False
    if word.lower() in _ABBREVIATIONS:
        return True
    # A single capital letter before the period is an initial ("J. Smith"), while a single
    # lowercase letter is just a token ("1000 g. SUMMARY" ends a sentence).
    if len(word) == 1 and word.isalpha() and word.isupper():
        return True
    return False
