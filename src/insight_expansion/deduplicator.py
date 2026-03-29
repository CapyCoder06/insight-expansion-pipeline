"""Deduplication of generated insights."""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Set, Optional


class Deduplicator:
    """Removes duplicate and near-duplicate insights."""

    SIMILARITY_THRESHOLD = 0.85

    @staticmethod
    def deduplicate(insights: List[Dict], seed_insights: List[Dict] = None) -> List[Dict]:
        if seed_insights is None:
            seed_insights = []

        # Build combined list: (insight, is_seed, order_index)
        combined: List[Tuple[Dict, bool, int]] = []
        # Expanded insights: is_seed=False, order = original index
        for idx, ins in enumerate(insights):
            combined.append((ins, False, idx))
        # Seed insights: is_seed=True, order = seed list index
        for idx, seed in enumerate(seed_insights):
            combined.append((seed, True, idx))

        n = len(combined)
        if n == 0:
            return []

        # Build similarity graph
        adj: List[List[int]] = [[] for _ in range(n)]
        texts = [item[0]["text"] for item in combined]
        for i in range(n):
            for j in range(i + 1, n):
                sim = SequenceMatcher(None, texts[i].lower(), texts[j].lower()).ratio()
                if sim > Deduplicator.SIMILARITY_THRESHOLD:
                    adj[i].append(j)
                    adj[j].append(i)

        # Signature-based deduplication: connect insights with same entity+metric+value
        signatures_map: Dict[Tuple[str, str, float], List[int]] = {}
        for idx, (insight, _, _) in enumerate(combined):
            sigs = Deduplicator._extract_signature(insight["text"])
            if sigs:
                # Normalize to list
                if not isinstance(sigs, list):
                    sigs = [sigs]
                for sig in sigs:
                    signatures_map.setdefault(sig, []).append(idx)

        # Connect all insights sharing the same signature
        for sig, indices in signatures_map.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx_i = indices[i]
                        idx_j = indices[j]
                        if idx_j not in adj[idx_i]:
                            adj[idx_i].append(idx_j)
                        if idx_i not in adj[idx_j]:
                            adj[idx_j].append(idx_i)

        # Find connected components (clusters)
        visited = [False] * n
        clusters: List[List[int]] = []
        for i in range(n):
            if not visited[i]:
                stack = [i]
                comp = []
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    comp.append(node)
                    for neigh in adj[node]:
                        if not visited[neigh]:
                            stack.append(neigh)
                clusters.append(comp)

        # Determine which indices to keep
        keep: Set[int] = set()
        for cluster in clusters:
            # If cluster contains any seed, keep all seed members; drop non-seeds
            cluster_seeds = [idx for idx in cluster if combined[idx][1]]
            if cluster_seeds:
                for idx in cluster_seeds:
                    keep.add(idx)
            else:
                # No seeds: pick the most complete insight (highest completeness score)
                best_idx = None
                best_score = -1
                for idx in cluster:
                    ins = combined[idx][0]
                    score = Deduplicator._completeness_score(ins)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_idx is not None:
                    keep.add(best_idx)

        # Separate kept seeds and non-seeds with their order indices
        kept_seeds: List[Tuple[int, Dict]] = []
        kept_expanded: List[Tuple[int, Dict]] = []
        for idx in keep:
            ins, is_seed, order = combined[idx]
            if is_seed:
                kept_seeds.append((order, ins))
            else:
                kept_expanded.append((order, ins))

        # Sort seeds by their order in seed_insights; expanded by original index
        kept_seeds.sort(key=lambda x: x[0])
        kept_expanded.sort(key=lambda x: x[0])

        # Concatenate: seeds first, then expanded
        result = [ins for _, ins in kept_seeds] + [ins for _, ins in kept_expanded]
        return result

    @staticmethod
    def _completeness_score(insight: Dict) -> int:
        """Score based on number of fields and metrics; more is better."""
        score = 0
        required = {"text", "dimensions", "metrics", "type_hint"}
        # Optional fields
        for key in insight:
            if key not in required:
                score += 1
        # More metrics is better
        score += len(insight.get("metrics", []))
        # Anomaly-specific bonuses
        if insight.get("type_hint") == "anomaly":
            if "issue" in insight:
                score += 1
            if "possible_cause" in insight:
                score += 2
        return score

    @staticmethod
    def _extract_signature(text: str) -> Optional[List[Tuple[str, str, float]]]:
        """
        Extract (entity, metric, value) signature(s) from insight text.
        Returns a list of signatures (multiple for multi-metric insights).
        Returns None if no signature can be extracted.
        """
        # Normalize text: remove parenthetical notes, extra spaces
        clean_text = re.sub(r'\s*\([^)]*\)', '', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        signatures = []

        # Pattern 1: "Entity metric is VALUE" e.g., "Africa margin is 11.3%" or "Africa profit is $88,872"
        pattern1 = r'^([A-Za-z\s]+?)\s+([a-z_]+)\s+is\s+(\$?[\d,]+(?:\.\d+)?%?)'
        match = re.search(pattern1, clean_text, re.IGNORECASE)
        if match:
            entity, metric, value_str = match.groups()
            entity = entity.strip()
            value = Deduplicator._parse_value(value_str)
            if value is not None:
                signatures.append((entity, metric, value))
                return signatures if len(signatures) > 1 else signatures[0] if signatures else None

        # Pattern 2a: "Entity has/achieves/generates ... metric of VALUE" e.g., "Africa has below average margin of 11.3%"
        # Also handles colon separator: "Africa has moderate margin: 11.3%"
        pattern2a = r'^([A-Za-z\s]+?)\s+(?:has|achieves|generates)\s+.*?([a-z_]+)\s*(?:of|:)\s*(\$?[\d,]+(?:\.\d+)?%?)'
        match = re.search(pattern2a, clean_text, re.IGNORECASE)
        if match:
            entity, metric, value_str = match.groups()
            entity = entity.strip()
            value = Deduplicator._parse_value(value_str)
            if value is not None:
                signatures.append((entity, metric, value))
                return signatures if len(signatures) > 1 else signatures[0] if signatures else None

        # Pattern 2b: "Entity metric of VALUE" (no has/achieves) e.g., "Africa revenue of $783,776"
        pattern2b = r'^([A-Za-z\s]+?)\s+([a-z_]+)\s*of\s*(\$?[\d,]+(?:\.\d+)?%?)'
        match = re.search(pattern2b, clean_text, re.IGNORECASE)
        if match:
            entity, metric, value_str = match.groups()
            entity = entity.strip()
            value = Deduplicator._parse_value(value_str)
            if value is not None:
                signatures.append((entity, metric, value))
                return signatures if len(signatures) > 1 else signatures[0] if signatures else None

        # Pattern 3: "Entity: metric1 VALUE1, metric2 VALUE2" (multi-metric)
        pattern3 = r'^([A-Za-z\s]+?):\s+([a-z_]+)\s+(\$?[\d,]*\d(?:\.\d+)?%?)(?:,\s*([a-z_]+)\s+([\$]?[\d,]*\d(?:\.\d+)?%?))?'
        match = re.search(pattern3, clean_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            entity = groups[0].strip()
            # First metric-value pair
            metric1, val_str1 = groups[1], groups[2]
            value1 = Deduplicator._parse_value(val_str1)
            if value1 is not None:
                signatures.append((entity, metric1, value1))
            # Optional second metric-value pair
            if len(groups) >= 5 and groups[3] and groups[4]:
                metric2, val_str2 = groups[3], groups[4]
                value2 = Deduplicator._parse_value(val_str2)
                if value2 is not None:
                    signatures.append((entity, metric2, value2))
            if signatures:
                return signatures

        # No pattern matched
        return None

    @staticmethod
    def _parse_value(value_str: str) -> Optional[float]:
        """Parse numeric value from string, handling currency and percentages."""
        if not value_str:
            return None
        # Remove currency symbols, commas
        cleaned = value_str.replace('$', '').replace(',', '').strip()
        # Handle percentage: 11.3% -> 11.3
        if cleaned.endswith('%'):
            cleaned = cleaned[:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None
