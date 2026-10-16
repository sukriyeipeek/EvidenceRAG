import math
import re
from collections import Counter


_WORD = re.compile(r"\w+")
# Turkish words are a stem plus suffixes ("faturalarımızın"), so terms are matched on their
# first five letters. Studies of Turkish search found this as good as real stemmers; the
# Snowball Turkish stemmer was no better on the evaluation sets and is inconsistent
# ("fatura" -> "fatur" but "faturalar" -> "fatura", "gün" -> "gün" but "günü" -> "g").
PREFIX_LENGTH = 5


class KeywordIndex:
    """BM25 ranking over word stems.

    Embeddings match meaning but blur exact tokens such as "E-1043" or "%14"; BM25
    rewards documents that contain the query's words, more so for rare words, and so
    complements the vector search.
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(counts.values()) for counts in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        document_frequency = Counter(term for counts in self.documents for term in counts)
        total = len(self.documents)
        # A word found in few documents says more about a match than a word found in all.
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def search(self, query, top_k=5):
        """Return (position, score) pairs for the best-matching texts, best first."""
        terms = [term for term in set(tokenize(query)) if term in self.idf]
        scores = []
        for position, counts in enumerate(self.documents):
            score = 0.0
            for term in terms:
                frequency = counts.get(term)
                if not frequency:
                    continue
                # Repeating a word helps less and less, and long texts are discounted so
                # they don't win just by containing more words.
                length_ratio = self.lengths[position] / self.average_length
                score += self.idf[term] * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                )
            if score > 0:
                scores.append((position, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]


def tokenize(text):
    # str.lower() turns "I" into "i" and "İ" into "i" plus a combining dot; Turkish
    # needs "ı" and "i".
    text = text.replace("I", "ı").replace("İ", "i").lower()
    return [word[:PREFIX_LENGTH] for word in _WORD.findall(text)]
