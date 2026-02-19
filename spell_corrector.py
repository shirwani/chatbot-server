from __future__ import annotations

from typing import Iterable

try:
    # Optional dependency providing a large English word list
    from wordfreq import wordlist as _wf_wordlist
except ImportError:  # pragma: no cover - handled gracefully at runtime
    _wf_wordlist = None


class SpellCorrector:
    """Simple spell corrector for standalone dictionary words in a string.

    - Only whole words that appear in the provided dictionary are corrected.
    - Punctuation and whitespace are preserved.
    - Case of the original word is preserved (lower, UPPER, Title).

    You can either pass an explicit *dictionary* iterable of words, or let
    :meth:`from_english_dictionary` build one from the optional ``wordfreq``
    package.
    """

    def __init__(self, dictionary: Iterable[str]):
        # Normalise dictionary to lowercase for matching
        self.dictionary = {w.lower() for w in dictionary}

    @classmethod
    def from_english_dictionary(
        cls,
        *,
        min_length: int = 2,
        max_words: int | None = None,
    ) -> "SpellCorrector":
        """Build a :class:`SpellCorrector` using a general English word list.

        This uses the optional ``wordfreq`` package if it is installed.

        Parameters
        ----------
        min_length:
            Ignore words shorter than this, to cut noise (default: 2).
        max_words:
            If set, only the ``max_words`` most frequent words from wordfreq
            are kept. This can improve performance.

        Raises
        ------
        RuntimeError
            If ``wordfreq`` is not installed.
        """
        wordlist_func = _wf_wordlist
        if wordlist_func is None:
            raise RuntimeError(
                "wordfreq is not installed. Install it with 'pip install wordfreq' "
                "or pass your own dictionary to SpellCorrector(...)."
            )

        words = list(wordlist_func("en"))
        if max_words is not None:
            words = words[:max_words]

        filtered = [w for w in words if len(w) >= min_length]
        return cls(filtered)

    # --- public API -----------------------------------------------------

    def fix_string(self, text: str) -> str:
        """Return *text* with misspelled dictionary words corrected.

        Words that are not close to any dictionary entry (edit distance > 2)
        are left unchanged.
        """
        tokens = self._tokenize(text)
        corrected_tokens: list[str] = []

        for token in tokens:
            if token.isalpha():
                corrected_tokens.append(self._correct_word_preserve_case(token))
            else:
                corrected_tokens.append(token)

        return "".join(corrected_tokens)

    # --- internal helpers -----------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split *text* into alphanumeric tokens and separator tokens.

        We can't just call ``split()`` because we must preserve spaces and
        punctuation in their original positions.
        """
        if not text:
            return []

        tokens: list[str] = []
        current = []
        is_alpha = text[0].isalpha()

        for ch in text:
            if ch.isalpha() == is_alpha:
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                current = [ch]
                is_alpha = ch.isalpha()
        if current:
            tokens.append("".join(current))
        return tokens

    def _correct_word_preserve_case(self, word: str) -> str:
        lower = word.lower()
        if lower in self.dictionary:
            return word

        candidate = self._best_candidate(lower)
        if candidate is None:
            return word

        return self._apply_case(word, candidate)

    def _best_candidate(self, word: str) -> str | None:
        """Return the closest dictionary word to *word* with edit distance <= 2."""
        best_word: str | None = None
        best_dist = 3  # we only care up to distance 2

        for w in self.dictionary:
            # small optimisation: skip obviously far words by length
            if abs(len(w) - len(word)) > 2:
                continue
            d = self._levenshtein(word, w)
            if d < best_dist:
                best_dist = d
                best_word = w
                if best_dist == 1:
                    # can't do better than 1 for our purposes
                    break

        return best_word

    @staticmethod
    def _apply_case(original: str, corrected_lower: str) -> str:
        if original.isupper():
            return corrected_lower.upper()
        if original[0].isupper() and original[1:].islower():
            return corrected_lower.capitalize()
        return corrected_lower

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between *a* and *b*.

        Implementation is optimised for short words (dictionary tokens).
        """
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        # Ensure a is the shorter string to use less memory
        if len(a) > len(b):
            a, b = b, a

        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current_row = [i]
            for j, cb in enumerate(b, start=1):
                insert_cost = current_row[j - 1] + 1
                delete_cost = previous_row[j] + 1
                replace_cost = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insert_cost, delete_cost, replace_cost))
            previous_row = current_row
        return previous_row[-1]


if __name__ == "__main__":
    # Demo using the built-in English dictionary if available, otherwise
    # fall back to a tiny manual list.
    if _wf_wordlist is not None:
        corrector = SpellCorrector.from_english_dictionary(min_length=2, max_words=50000)
    else:
        dictionary = ["store", "hours", "waterproof", "jacket", "under", "hiring"]
        corrector = SpellCorrector(dictionary)

    s = "Wat ar your stoer hurs? Are yu hireing?"
    print(corrector.fix_string(s))
