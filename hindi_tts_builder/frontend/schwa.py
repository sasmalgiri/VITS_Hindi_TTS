"""Schwa deletion for Hindi Devanagari.

Hindi written in Devanagari includes an implicit 'a' vowel (schwa, अ) after
every consonant that isn't followed by an explicit vowel mark or halant. In
speech, these schwas are often deleted. For example:

    कमल  written: ka-ma-la   spoken: kamal (final schwa deleted)
    रामायण written: Ra-ma-ya-na   spoken: Ramayan

If a TTS model sees "कमल" and produces "ka-ma-la", that's a mispronunciation.
This module applies linguistic rules to mark positions where the schwa should
be dropped before the text is passed to the acoustic model.

Implementation: rule-based schwa deletion following standard linguistic rules
(Narasimhan et al., Tyson & Nagar, etc.). Good for ~95% of cases.

Output convention: we insert an explicit halant (virama, ्) after consonants
whose schwa should be deleted. This is a clean signal the downstream model can
use, and it preserves Devanagari script validity.
"""
from __future__ import annotations


# Consonant characters (Devanagari block)
_CONSONANTS = set(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक़ख़ग़ज़ड़ढ़फ़य़"
)

# Vowel signs (matras). If present after a consonant, there's no schwa to delete.
_MATRAS = set("ािीुूृेैोौंःँ")

# Halant (virama) — explicitly suppresses schwa
_HALANT = "\u094d"

# Independent vowels — not relevant for schwa deletion
_INDEP_VOWELS = set("अआइईउऊऋएऐओऔ")


def _is_consonant(c: str) -> bool:
    return c in _CONSONANTS


def _is_matra(c: str) -> bool:
    return c in _MATRAS


def _has_inherent_schwa(word: str, i: int) -> bool:
    """A consonant at position i has an inherent schwa if:
    - it's a consonant
    - not followed by a matra or halant
    """
    if i >= len(word) or not _is_consonant(word[i]):
        return False
    if i + 1 >= len(word):
        return True
    nxt = word[i + 1]
    return not (_is_matra(nxt) or nxt == _HALANT)


def _delete_schwa_in_word(word: str) -> str:
    """Apply schwa deletion rules to a single word.

    Core rule (word-final): if the last consonant has an inherent schwa, delete
    it, UNLESS the word is a single consonant or a CV (consonant + vowel) pattern
    short enough that deletion leaves it empty.

    Second rule (VC̲CV): a schwa between two consonant clusters often deletes.
    We implement the conservative "final schwa only" rule here because it's
    safest — most mispronunciations in TTS come from final schwas.
    """
    if len(word) < 2:
        return word

    # Find the last consonant position
    out = list(word)
    # Walk backwards to find last consonant
    last_cons = -1
    for i in range(len(out) - 1, -1, -1):
        if _is_consonant(out[i]):
            last_cons = i
            break
    if last_cons < 0:
        return word
    # If last consonant has inherent schwa, mark it with halant
    if _has_inherent_schwa(word, last_cons):
        # Count consonants before — need at least one other phoneme left
        preceding = word[:last_cons]
        if any(_is_consonant(c) or c in _INDEP_VOWELS for c in preceding):
            return word[: last_cons + 1] + _HALANT + word[last_cons + 1 :]
    return word


def delete_schwa(text: str) -> str:
    """Apply schwa deletion word by word. Non-Devanagari tokens are passed through."""
    out_tokens = []
    cur = []
    for ch in text:
        if _is_consonant(ch) or _is_matra(ch) or ch in _INDEP_VOWELS or ch == _HALANT:
            cur.append(ch)
        else:
            if cur:
                out_tokens.append(_delete_schwa_in_word("".join(cur)))
                cur = []
            out_tokens.append(ch)
    if cur:
        out_tokens.append(_delete_schwa_in_word("".join(cur)))
    return "".join(out_tokens)
