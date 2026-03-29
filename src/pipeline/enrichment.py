"""
Enrichment module for generating semantic variations of documents.

Creates multiple wording variations while preserving exact meaning and facts.
"""

from typing import List, Dict, Any, TYPE_CHECKING
import random
import re

if TYPE_CHECKING:
    from .config import Config
from .validator import semantic_similarity


class DocumentEnricher:
    """Generates semantic variations of documents."""

    # Controlled synonym vocabulary - synonyms must be contextually equivalent
    SYNONYMS = {
        # Verbs
        "has": ["possesses", "exhibits", "shows", "displays", "demonstrates"],
        "have": ["possess", "exhibit", "show", "display", "demonstrate"],
        "leads": ["ranks first", "is top", "is highest", "topped"],
        "increased": ["rose", "grew", "climbed", "went up", "ascended"],
        "decreased": ["fell", "dropped", "declined", "went down", "diminished"],
        "causes": ["results in", "leads to", "triggers", "induces", "contributes to"],
        "achieved": ["reached", "attained", "obtained", "recorded", "posted"],
        "generate": ["produce", "yield", "deliver", "result in"],
        "losing": ["experiencing losses", "in the red", "showing negative", "with negative"],

        # Adjectives
        "highest": ["maximum", "peak", "top", "greatest", "best"],
        "lowest": ["minimum", "bottom", "worst", "lowest"],
        "negative": ["below zero", "in the red", "loss", "deficit"],
        "positive": ["above zero", "profitable", "in the black"],

        # Connectors
        "and": ["&", "as well as", "plus", "along with"],
        "with": ["including", "featuring", "having", "containing"],
        "of": ["from", "valued at", "totaling"],

        # Nouns
        "profit": ["earnings", "income", "gains", "net income"],
        "margin": ["profitability", "return", "rate"],
        "sales": ["revenue", "turnover", "income"],
    }

    # Sentence restructuring patterns (for FACTs)
    RESTRUCTURES = [
        "{subject} {verb} {object}",
        "The data shows that {subject} {verb} {object}",
        "Analysis indicates {subject} {verb} {object}",
        "{object} is {verb} by {subject}",
    ]

    # Query-aware patterns - convert facts to searchable questions
    QUERY_PATTERNS = {
        'fact': {
            'highest': [
                # Short, high-similarity patterns (pass >=0.6 filter)
                "Highest {metric}?",
                "Has highest {metric}?",
                # Medium length
                "Which {dimension} has highest {metric}?",
            ],
            'lowest': [
                "Lowest {metric}?",
                "Has lowest {metric}?",
                "Which {dimension} has lowest {metric}?",
            ],
            'comparison': [
                "How does {metric} compare?",
                "Compare {metric}",
            ],
        },
        'trend': {
            'increasing': [
                "Is {metric} increasing?",
                "Has {metric} increased?",
                "Increased {metric}?",
                # Include time phrase if present in original (will be filtered by similarity)
                "Has {metric} increased year over year?",
                "Increased {metric} year over year?",
            ],
            'decreasing': [
                "Is {metric} decreasing?",
                "Has {metric} decreased?",
                "Decreased {metric}?",
                "Has {metric} decreased year over year?",
                "Decreased {metric} year over year?",
            ],
            'general': [
                "What is {metric} trend?",
                "How has {metric} changed?",
            ]
        },
        'anomaly': {
            'negative': [
                "Negative {metric}?",
                "{metric} loss?",
                "{metric} anomaly?",
            ],
            'outlier': [
                "Unusual {metric}?",
                "{metric} outlier?",
            ]
        },
        'metric': {
            'definition': [
                "What is {metric}?",
                "Define {metric}",
                "How to calculate {metric}?",
            ]
        }
    }

    def __init__(self, num_variations: int = None, config: 'Config' = None):
        """
        Initialize enricher.

        Args:
            num_variations: Target number of variations to generate (overrides config if specified)
            config: Configuration object (loads from config.yaml if None)
        """
        # Lazy import to avoid circular dependency
        if config is None:
            from .config import get_config
            config = get_config()

        enrichment_config = config.get('enrichment', {})
        if num_variations is None:
            num_variations = enrichment_config.get('variations_to_generate', 8)

        self.num_variations = min(10, max(5, num_variations))

    def expand_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate multiple semantic variations of a document.

        Args:
            document: Input document with 'text' and 'metadata'

        Returns:
            List of variation documents (5-10 variations)
        """
        if not document or "text" not in document:
            return []

        original_text = document["text"]
        metadata = document["metadata"]
        doc_type = metadata.get("type", "fact")

        # Generate variations based on document type
        if doc_type == "fact":
            variations = self._generate_fact_variations(original_text, metadata)
        elif doc_type == "trend":
            variations = self._generate_trend_variations(original_text, metadata)
        elif doc_type == "anomaly":
            variations = self._generate_anomaly_variations(original_text, metadata)
        elif doc_type == "metric":
            variations = self._generate_metric_variations(original_text, metadata)
        elif doc_type == "question":
            variations = self._generate_question_variations(original_text, metadata)
        else:
            # Fallback: simple synonym replacement
            variations = self._generate_simple_variations(original_text)

        # Ensure we have metadata attached to each variation
        result = []
        for var_text in variations[:self.num_variations]:
            result.append({
                "text": var_text,
                "metadata": metadata.copy()  # Preserve exact same metadata
            })

        # Always include original as first variation (guarantee minimum quality)
        if not any(v["text"] == original_text for v in result):
            result.insert(0, {
                "text": original_text,
                "metadata": metadata.copy()
            })

        # If we still have less than self.num_variations, add generic framing variations
        if len(result) < self.num_variations:
            # Extract core content (body) from original
            lines = original_text.split('\n')
            header = lines[0] if lines else ""
            core_content = ""
            for line in lines[1:]:
                stripped = line.strip()
                if stripped:
                    core_content = stripped
                    break
            if not core_content:
                core_content = original_text

            # Generic frames that preserve meaning (no new facts)
            generic_frames = [
                f"Analysis reveals that {core_content}",
                f"Data shows {core_content}",
                f"We observe: {core_content}",
                f"The facts indicate {core_content}",
                f"Report states: {core_content}",
                f"Summary: {core_content}",
                f"According to analysis, {core_content}",
                f"Findings suggest {core_content}",
            ]

            needed = self.num_variations - len(result)
            for frame in generic_frames[:needed]:
                full_doc = f"{header}\n\n{frame}"
                result.append({
                    "text": full_doc,
                    "metadata": metadata.copy()
                })

        return result[:self.num_variations]

    def _generate_fact_variations(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate variations for FACT documents."""
        lines = text.split('\n')
        if len(lines) < 2:
            return [text]

        header = lines[0]
        # Find first non-empty line after header for content
        core_content = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                core_content = stripped
                break

        if not core_content:
            return [text]

        variations = set()

        # Always include full core content
        variations.add(core_content)

        # 1. Synonym substitution (global - all applicable words)
        variations.add(self._replace_synonyms_global(core_content))

        # 2. Reporting verb prefix
        for prefix in ["Analysis shows", "Data indicates", "Evidence suggests", "Findings reveal"]:
            variations.add(f"{prefix} that {core_content.lower()}")

        # 3. Metric-focused framing
        metric = metadata.get("metrics", [""])[0] if metadata.get("metrics") else ""
        if metric:
            variations.add(f"The {metric} reveals {core_content}")

        # 4. Dimension-focused framing
        dimension = metadata.get("dimensions", [""])[0] if metadata.get("dimensions") else ""
        if dimension:
            variations.add(f"For {dimension}, {core_content}")

        # 5. Passive voice transformation (if applicable)
        if " has " in core_content or " have " in core_content:
            passive = core_content.replace(" has ", " is characterized by ").replace(" have ", " are characterized by ")
            variations.add(passive)

        # 6. Data assertion framing
        variations.add(f"It is clear that {core_content}")

        # 7. "What we know" framing
        variations.add(f"We observe that {core_content}")

        # 8. Simple reordering (subject at end)
        words = core_content.split()
        if len(words) > 3:
            # Move last part to front
            second_half = " ".join(words[-2:])
            first_half = " ".join(words[:-2])
            variations.add(f"{second_half}, {first_half}")

        # 9. QUERY-AWARE VARIATIONS - Generate searchable question patterns
        query_variations = self._generate_query_variations(core_content, metadata, 'fact')
        variations.update(query_variations)

        # Format with header and deduplicate
        cleaned = []
        for v in variations:
            v = v.strip()
            if v and len(v) > 5:  # Not empty
                full_doc = f"{header}\n\n{v}"
                if full_doc not in cleaned:
                    cleaned.append(full_doc)

        return cleaned if cleaned else [text]

    def _generate_trend_variations(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate variations for TREND documents."""
        lines = text.split('\n')
        if len(lines) < 2:
            return [text]

        header = lines[0]
        # Find first non-empty line after header
        core_content = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                core_content = stripped
                break

        if not core_content:
            return [text]
        variations = set()

        variations.add(core_content)
        variations.add(self._replace_synonyms_global(core_content))

        # Multiple framing approaches
        prefixes = [
            "Over the observed period,",
            "Looking at the time series,",
            "Analysis of the trend shows",
            "The trajectory demonstrates",
            "Data reveals a pattern where"
        ]

        for prefix in prefixes:
            variations.add(f"{prefix} {core_content.lower()}")

        # Direction emphasis from metadata
        trend_direction = metadata.get("trend", "increasing")
        if trend_direction in ["increasing", "upward", "rising"]:
            for phrase in ["an upward trend", "growth", "increase"]:
                variations.add(f"The data shows {phrase}: {core_content}")
        else:
            for phrase in ["a downward trend", "decline", "decrease"]:
                variations.add(f"The data shows {phrase}: {core_content}")

        # Year-over-year framing
        variations.add(f"Year-over-year analysis indicates {core_content}")

        # CAGR emphasis if present
        if "cagr" in core_content.lower():
            variations.add(f"The compound annual growth rate (CAGR) is notable: {core_content}")

        # Query-aware variations
        query_variations = self._generate_query_variations(core_content, metadata, 'trend')
        variations.update(query_variations)

        # Format
        cleaned = []
        for v in variations:
            v = v.strip()
            if v and len(v) > 5:
                full_doc = f"{header}\n\n{v}"
                if full_doc not in cleaned:
                    cleaned.append(full_doc)

        return cleaned if cleaned else [text]

    def _generate_anomaly_variations(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate variations for ANOMALY documents."""
        lines = text.split('\n')
        if len(lines) < 2:
            return [text]

        header = lines[0]
        # Find first non-empty line after header
        core_content = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                core_content = stripped
                break

        if not core_content:
            return [text]
        variations = set()

        variations.add(core_content)
        variations.add(self._replace_synonyms_global(core_content))

        # Issue framing templates
        templates = [
            "An anomaly has been detected: {content}",
            "Investigation reveals an unexpected pattern: {content}",
            "A concerning issue exists: {content}",
            "Data shows an outlier: {content}",
            "An unexpected deviation is present: {content}",
            "Our analysis identified a problem: {content}",
        ]

        for template in templates:
            variations.add(template.format(content=core_content))

        # Cause integration
        cause = metadata.get("possible_cause", "")
        if cause:
            variations.add(f"{core_content} This appears to be caused by {cause}")

        # Query-aware variations
        query_variations = self._generate_query_variations(core_content, metadata, 'anomaly')
        variations.update(query_variations)

        # Format
        cleaned = []
        for v in variations:
            v = v.strip()
            if v and len(v) > 5:
                full_doc = f"{header}\n\n{v}"
                if full_doc not in cleaned:
                    cleaned.append(full_doc)

        return cleaned if cleaned else [text]

    def _generate_metric_variations(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate variations for METRIC documents."""
        lines = text.split('\n')
        if len(lines) < 2:
            return [text]

        header = lines[0]
        # Find first non-empty line after header
        core_content = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                core_content = stripped
                break

        if not core_content:
            return [text]

        variations = set()
        variations.add(core_content)
        variations.add(self._replace_synonyms_global(core_content))

        # Query-aware variations (for metrics, this adds definition questions)
        query_variations = self._generate_query_variations(core_content, metadata, 'metric')
        variations.update(query_variations)

        # For metrics, keep variation minimal to preserve precision
        cleaned = []
        for v in variations:
            v = v.strip()
            if v and len(v) > 5:
                full_doc = f"{header}\n\n{v}"
                if full_doc not in cleaned:
                    cleaned.append(full_doc)

        return cleaned if cleaned else [text]

    def _generate_question_variations(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Generate variations for QUESTION documents."""
        lines = text.split('\n')
        if len(lines) < 3:  # Need Q and A
            return [text]

        header = lines[0]
        # Extract Q and A lines
        question_line = None
        answer_line = None

        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Q:"):
                question_line = stripped
            elif stripped.startswith("A:"):
                answer_line = stripped

        if not question_line or not answer_line:
            return [text]

        q_text = question_line[2:].strip()
        a_text = answer_line[2:].strip()

        variations = set()
        variations.add(f"Q: {q_text}\nA: {a_text}")
        variations.add(f"Q: {q_text}\nThe answer is: {a_text}")
        variations.add(f"How would you answer: {q_text}\nAnswer: {a_text}")
        variations.add(f"Question: {q_text}\nResponse: {a_text}")

        cleaned = []
        for v in variations:
            v = v.strip()
            if v and len(v) > 5:
                full_doc = f"{header}\n\n{v}"
                if full_doc not in cleaned:
                    cleaned.append(full_doc)

        return cleaned if cleaned else [text]

    def _extract_query_entities(self, text: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract key entities from document for query pattern filling.
        Returns dict with keys: dimension, metric, entity, value, trend_type
        """
        entities = {
            'dimension': '',
            'metric': '',
            'entity': '',
            'value': '',
            'trend_type': ''
        }

        # Extract from metadata first (most reliable)
        dimensions = metadata.get("dimensions", [])
        metrics = metadata.get("metrics", [])
        trend = metadata.get("trend", "")

        if dimensions:
            entities['dimension'] = dimensions[0]
        if metrics:
            entities['metric'] = metrics[0]
        if trend:
            entities['trend_type'] = trend.lower()

        # Extract from text as fallback using regex patterns
        text_lower = text.lower()

        # Try to extract numeric values (e.g., $664K, 14%, 24%)
        value_patterns = [
            r'\$[\d,]+K?',  # $664K, $2M
            r'\d+(?:\.\d+)?%',  # 14%, 24.5%
            r'-?\d+(?:\.\d+)?[\w$]*'  # General numbers
        ]
        for pattern in value_patterns:
            match = re.search(pattern, text)
            if match:
                entities['value'] = match.group()
                break

        # Extract entity (subject) - look for capitalized nouns or proper nouns
        # Simple heuristic: words after the first verb or after "has", "shows", etc.
        if not entities['entity']:
            # Try to find subject by looking for pattern "X has/shows/leads"
            subject_patterns = [
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:has|shows|leads|demonstrates)',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:has|shows|leads|demonstrates)'
            ]
            for pattern in subject_patterns:
                match = re.search(pattern, text)
                if match:
                    entities['entity'] = match.group(1).lower()
                    break

        return entities

    def _generate_query_variations(self, core_content: str, metadata: Dict[str, Any], doc_type: str) -> List[str]:
        """
        Generate user-intent query patterns (NOT answers).

        Generates realistic search queries that users would type when seeking this information.
        - Does NOT include specific entity names from the document
        - Does NOT include numeric values
        - Generalizes to ask about the pattern/concept, not the specific instance
        - Adds diversity: short, natural language, keyword-based
        """
        if doc_type not in self.QUERY_PATTERNS:
            return []

        entities = self._extract_query_entities(core_content, metadata)
        patterns_by_category = self.QUERY_PATTERNS[doc_type]

        queries = set()

        # DON'T include specific entities or numeric values from this document
        # Use only generic dimension/metric placeholders

        # Determine which pattern category to use based on content
        if doc_type == 'fact':
            # Detect if it's about highest/best
            if any(word in core_content.lower() for word in ['highest', 'top', 'best', 'maximum', 'peak', 'greatest']):
                category = 'highest'
                patterns = patterns_by_category.get(category, [])
                for pattern in patterns:
                    query = pattern.format(
                        dimension=entities['dimension'] or 'category',
                        metric=entities['metric'] or 'profit',
                    )
                    queries.add(query)
            elif any(word in core_content.lower() for word in ['lowest', 'worst', 'bottom', 'minimum']):
                category = 'lowest'
                patterns = patterns_by_category.get(category, [])
                for pattern in patterns:
                    query = pattern.format(
                        dimension=entities['dimension'] or 'category',
                        metric=entities['metric'] or 'profit',
                    )
                    queries.add(query)
            elif 'compare' in core_content.lower() or 'across' in core_content.lower():
                category = 'comparison'
                patterns = patterns_by_category.get(category, [])
                for pattern in patterns:
                    query = pattern.format(
                        dimension=entities['dimension'] or 'category',
                        metric=entities['metric'] or 'profit',
                    )
                    queries.add(query)
            # Note: 'value' category intentionally omitted to avoid entity-specific questions

        elif doc_type == 'trend':
            # Detect trend direction
            trend_type = entities['trend_type']
            if trend_type in ['increasing', 'upward', 'rising', 'growing']:
                category = 'increasing'
            elif trend_type in ['decreasing', 'downward', 'declining', 'falling']:
                category = 'decreasing'
            else:
                category = 'general'

            patterns = patterns_by_category.get(category, patterns_by_category['general'])
            for pattern in patterns:
                query = pattern.format(metric=entities['metric'] or 'sales')
                queries.add(query)

        elif doc_type == 'anomaly':
            # Detect anomaly type
            if 'negative' in core_content.lower() or 'loss' in core_content.lower() or 'below zero' in core_content.lower():
                category = 'negative'
            else:
                category = 'outlier'

            patterns = patterns_by_category.get(category, patterns_by_category['outlier'])
            for pattern in patterns:
                query = pattern.format(
                    dimension=entities['dimension'] or 'category',
                    metric=entities['metric'] or 'profit',
                )
                queries.add(query)

        elif doc_type == 'metric':
            patterns = patterns_by_category['definition']
            for pattern in patterns:
                query = pattern.format(metric=entities['metric'] or 'metric')
                queries.add(query)

        # Filter by semantic similarity to ensure query form is semantically related
        # Threshold 0.6 matches the drift test requirement
        filtered_queries = set()
        similarity_threshold = 0.6
        for q in queries:
            sim = semantic_similarity(core_content, q)
            if sim >= similarity_threshold:
                filtered_queries.add(q)

        return list(filtered_queries)

    def _generate_simple_variations(self, text: str) -> List[str]:
        """Fallback simple variations."""
        return [self._replace_synonyms_global(text)]

    def _restructure_sentence(self, text: str) -> str:
        """Apply a simple restructuring if pattern matches."""
        # Very simple: subject-verb-object rearrangement for certain patterns
        words = text.split()
        if len(words) >= 3:
            # "X has Y" -> "Y is possessed by X"
            if "has" in words:
                idx = words.index("has")
                if idx > 0 and idx < len(words) - 1:
                    subj = " ".join(words[:idx])
                    obj = " ".join(words[idx+1:])
                    return f"{obj} is possessed by {subj}"
        return text  # Return original if no clear pattern

    def _replace_synonyms_global(self, text: str) -> str:
        """Apply multiple synonym replacements across the text."""
        result = text
        # Apply up to 3 synonym replacements to create a genuinely different but equivalent text
        replacements_made = 0
        max_replacements = 3

        for original, synonyms in self.SYNONYMS.items():
            if replacements_made >= max_replacements:
                break
            # Use a regex to find whole word matches
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, result, re.IGNORECASE):
                # Replace with a random synonym
                replacement = random.choice(synonyms)
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                replacements_made += 1

        return result

    def _rephrase_question(self, question: str) -> str:
        """Rephrase a question (simple transformations)."""
        # Can add more sophisticated rephrasing later
        return question  # For now, return as-is


def expand_document(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand a document into multiple semantic variations.

    Args:
        document: Input document with 'text' and 'metadata'

    Returns:
        List of 5-10 variation documents with preserved metadata
    """
    enricher = DocumentEnricher()
    return enricher.expand_document(document)


def select_top_variations(variations: List[Dict[str, Any]], top_k: int = 2,
                          similarity_threshold: float = 0.80) -> List[Dict[str, Any]]:
    """
    Select top k highest quality variations with deduplication.

    Args:
        variations: List of variation documents
        top_k: Number of top variations to return (default 2)
        similarity_threshold: Semantic similarity threshold for near-duplicate detection (default 0.85)

    Returns:
        List of top k semantically distinct variations (always includes first/original if present)
    """
    if not variations:
        return []

    if len(variations) <= top_k:
        return variations

    # Score variations based on quality heuristics
    scored = []
    for i, var in enumerate(variations):
        score = 0
        text = var.get("text", "")

        # Length quality (optimal ~50-500 chars)
        length = len(text)
        if 50 <= length <= 500:
            score += 2
        elif length > 20:
            score += 1

        # Completeness (has text)
        if text and len(text.strip()) > 0:
            score += 1

        # Avoid obvious errors
        if "  " in text:  # Multiple spaces
            score -= 1
        if not any(c.isalpha() for c in text):  # No letters
            score -= 2

        # Prioritize original (first variation)
        if i == 0:
            score += 3

        scored.append((score, var))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by semantic similarity
    selected = []
    for _, var in scored:
        if len(selected) >= top_k:
            break

        text = var.get("text", "").strip().lower()
        if not text:
            continue

        # Check if this variation is too similar to any already selected
        is_duplicate = False
        for selected_var in selected:
            selected_text = selected_var.get("text", "").strip().lower()
            # Simple text overlap check first (fast path)
            if text == selected_text:
                is_duplicate = True
                break
            # Semantic similarity check
            similarity = semantic_similarity(text, selected_text)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(var)

    # If we have fewer than top_k due to deduplication, fill with highest scored remaining
    if len(selected) < top_k:
        for _, var in scored:
            if len(selected) >= top_k:
                break
            if var not in selected:
                selected.append(var)

    return selected


def enrich_documents(documents: List[Dict[str, Any]], select_top: int = 3) -> List[Dict[str, Any]]:
    """
    Enrich a list of documents by expanding each into variations and selecting top ones.

    Args:
        documents: List of input documents
        select_top: Number of top variations to keep per document (default 3)

    Returns:
        List of enriched document variations
    """
    enriched = []
    for doc in documents:
        variations = expand_document(doc)
        top_variations = select_top_variations(variations, top_k=select_top)
        enriched.extend(top_variations)

    return enriched
