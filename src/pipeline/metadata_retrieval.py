from typing import Dict, List, Any
from .validator import semantic_similarity


class QueryExpander:
    def __init__(self, global_synonyms: Dict[str, List[str]]):
        self.global_synonyms = global_synonyms

    def expand(self, query: str) -> List[str]:
        """Expand query by replacing terms with synonyms. Returns list of query variants."""
        variants = [query]
        words = query.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.global_synonyms:
                for synonym in self.global_synonyms[word_lower]:
                    new_words = words.copy()
                    new_words[i] = synonym
                    variants.append(' '.join(new_words))
        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)
        return unique_variants


class MetadataQueryMatcher:
    def similarity(self, query: str, metadata_queries: List[str]) -> float:
        """Compute maximum semantic similarity between query and any metadata query."""
        if not metadata_queries:
            return 0.0
        scores = [semantic_similarity(query, meta_q) for meta_q in metadata_queries]
        return max(scores) if scores else 0.0


class HybridScorer:
    def __init__(self, weights: Dict[str, float], bonus_config: Dict[str, Any] = None):
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            # Normalize to 1.0
            self.weights = {k: v/total for k, v in weights.items()}
        else:
            self.weights = weights

        # Default bonus configuration
        self.bonus_config = {
            'enabled': False,
            'threshold': 0.7,
            'boost_factor': 1.5
        }
        if bonus_config:
            self.bonus_config.update(bonus_config)

        # Compute maximum possible raw score for normalization
        # When all similarities = 1.0 and metadata qualifies for boost, raw_max = 1 + w_meta*(boost-1)
        self._max_raw_score = 1.0  # default when no boost
        if self.bonus_config.get('enabled', False):
            boost = self.bonus_config.get('boost_factor', 1.5)
            if boost > 1:
                w_meta = self.weights.get('metadata_query', 0.0)
                self._max_raw_score = 1.0 + w_meta * (boost - 1)

    def score(self, query: str, document: Dict[str, Any],
              expanded_variants: List[str], meta_matcher: MetadataQueryMatcher) -> float:
        text = document.get('text', '')
        if not text:
            return 0.0
        # 1. text similarity
        text_sim = semantic_similarity(query, text)
        # 2. metadata query similarity
        metadata_queries = document.get('metadata', {}).get('queries', [])
        meta_sim = meta_matcher.similarity(query, metadata_queries)
        # 3. expanded query similarity (max over variants)
        if expanded_variants:
            expanded_scores = [semantic_similarity(variant, text) for variant in expanded_variants]
            expanded_sim = max(expanded_scores)
        else:
            expanded_sim = 0.0

        # Apply metadata similarity bonus if configured and threshold exceeded
        w = self.weights
        meta_score = w['metadata_query'] * meta_sim
        if (self.bonus_config.get('enabled', False) and
            meta_sim >= self.bonus_config.get('threshold', 0.7)):
            boost_factor = self.bonus_config.get('boost_factor', 1.5)
            meta_score *= boost_factor

        # Combine raw score
        raw_score = (w['text'] * text_sim +
                     meta_score +
                     w['expanded'] * expanded_sim)

        # Normalize to [0, 1] using precomputed max raw score
        normalized_score = raw_score / self._max_raw_score

        # Clamp to [0, 1] for numerical safety (should be already within range)
        return max(0.0, min(1.0, normalized_score))
