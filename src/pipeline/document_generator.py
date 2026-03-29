"""
Document generator that applies templates to insights.
"""

from typing import Dict, Any, Optional
import json
from .types import DocumentType
from .templates import TEMPLATES
from .template_mapper import TemplateMapper


class DocumentGenerator:
    """Generates standardized documents from structured insights."""

    def __init__(self):
        """Initialize the document generator."""
        self.mapper = TemplateMapper()

    def generate_document(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a document from an insight.

        Args:
            insight: Structured insight with keys:
                - text (required): The main insight text
                - dimensions (optional): List of dimension names
                - metrics (optional): List of metric names
                - type_hint (optional): Preferred document type
                - Additional fields specific to document type

        Returns:
            Dictionary with:
                - text: Rendered document text (for embedding)
                - metadata: Dictionary with type, dimensions, metrics, etc.
        """
        # Map to document type
        doc_type = self.mapper.map_insight(insight)

        # Extract common fields
        text = insight.get("text", "")
        dimensions = insight.get("dimensions", [])
        metrics = insight.get("metrics", [])

        # Build template context based on document type
        context = self._build_context(doc_type, insight)

        # Render the document
        template = TEMPLATES[doc_type]
        document_text = template.format(**context)

        # Build metadata
        metadata = {
            "type": doc_type.value,
            "dimensions": dimensions,
            "metrics": metrics,
            "title": self._extract_title(text, doc_type),
            # Ranking signals
            "importance": self._infer_importance(insight, doc_type, text),
            "confidence": self._calculate_initial_confidence(insight, doc_type),
            "source": insight.get("source", "EDA")
        }

        # Add type-specific metadata
        if doc_type in [DocumentType.FACT, DocumentType.TREND]:
            metadata["grain"] = insight.get("grain", self._infer_grain(dimensions))
        if doc_type == DocumentType.TREND:
            metadata["dimension"] = insight.get("dimension", dimensions[0] if dimensions else "")
            metadata["trend"] = insight.get("trend", "increasing")
        elif doc_type == DocumentType.ANOMALY:
            metadata["dimension"] = insight.get("dimension", dimensions[0] if dimensions else "")
            metadata["metric"] = insight.get("metric", metrics[0] if metrics else "")
            metadata["issue"] = insight.get("issue", "unexpected behavior")
            metadata["possible_cause"] = insight.get("possible_cause", "requires investigation")
        elif doc_type == DocumentType.METRIC:
            metadata["metric"] = insight.get("metric", metrics[0] if metrics else "")
            metadata["definition"] = insight.get("definition", text)
            metadata["formula"] = insight.get("formula", "")
            metadata["interpretation"] = insight.get("interpretation", "")
        elif doc_type == DocumentType.QUESTION:
            metadata["question"] = insight.get("question", text)
            metadata["answer"] = insight.get("answer", "")

        return {
            "text": document_text.strip(),
            "metadata": metadata
        }

    def _build_context(self, doc_type: DocumentType, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Build template context from insight."""
        context = {
            "title": self._extract_title(insight.get("text", ""), doc_type),
            "description": insight.get("text", ""),
            "dimensions": ", ".join(insight.get("dimensions", [])),
            "metrics": ", ".join(insight.get("metrics", [])),
        }

        # Add type-specific fields
        if doc_type in [DocumentType.FACT, DocumentType.TREND]:
            context["grain"] = insight.get("grain", self._infer_grain(insight.get("dimensions", [])))
        if doc_type == DocumentType.TREND:
            context["dimension"] = insight.get("dimension", insight.get("dimensions", [""])[0] if insight.get("dimensions") else "")
            context["trend"] = insight.get("trend", "increasing")
        elif doc_type == DocumentType.ANOMALY:
            context["dimension"] = insight.get("dimension", insight.get("dimensions", [""])[0] if insight.get("dimensions") else "")
            context["metric"] = insight.get("metric", insight.get("metrics", [""])[0] if insight.get("metrics") else "")
            context["issue"] = insight.get("issue", "Unexpected deviation detected")
            context["possible_cause"] = insight.get("possible_cause", "Requires further investigation")
        elif doc_type == DocumentType.METRIC:
            metric_name = insight.get("metric", insight.get("metrics", [""])[0] if insight.get("metrics") else "Metric")
            context["metric"] = metric_name
            context["definition"] = insight.get("definition", insight.get("text", ""))
            context["formula"] = insight.get("formula", "")
            context["interpretation"] = insight.get("interpretation", "")
        elif doc_type == DocumentType.QUESTION:
            context["question"] = insight.get("question", insight.get("text", ""))
            context["answer"] = insight.get("answer", "")

        return context

    def _infer_importance(self, insight: Dict[str, Any], doc_type: DocumentType, text: str) -> str:
        """
        Infer importance level from insight content and type.

        Heuristics:
        - High: top performers (highest, best), anomalies, key metrics
        - Medium: trends, general facts, questions
        - Low: minor observations, average values
        """
        text_lower = text.lower()

        # Anomalies are typically high importance (problems to address)
        if doc_type == DocumentType.ANOMALY:
            return "high"

        # Check for high-importance indicators
        high_indicators = ["highest", "top", "best", "maximum", "peak", "critical", "key", "major"]
        if any(indicator in text_lower for indicator in high_indicators):
            return "high"

        # Check for low-importance indicators
        low_indicators = ["lowest", "worst", "minimum", "minor", "average", "typical", "some", "moderate"]
        if any(indicator in text_lower for indicator in low_indicators):
            return "low"

        # Anomalies, metrics with definitions tend to be medium
        if doc_type in [DocumentType.METRIC, DocumentType.TREND, DocumentType.QUESTION]:
            return "medium"

        # Default for FACT without strong indicators
        return "medium"

    def _calculate_initial_confidence(self, insight: Dict[str, Any], doc_type: DocumentType) -> float:
        """
        Calculate initial confidence score (0-1) for the document.

        Factors:
        - Complete insight (has all required fields)
        - Specific metrics/dimensions
        - Numeric values present
        - Document type clarity
        """
        score = 0.5  # Base confidence

        # Insight completeness
        required_fields = ["text", "dimensions", "metrics"]
        for field in required_fields:
            if field in insight and insight[field]:
                score += 0.1

        # Has type_hint = clear classification
        if "type_hint" in insight:
            score += 0.1

        # Has numeric values in metrics (data-driven)
        if insight.get("metrics"):
            score += 0.1

        # Type-specific adjustments
        if doc_type == DocumentType.ANOMALY:
            # Anomalies with possible_cause are more confident
            if "possible_cause" in insight and insight["possible_cause"]:
                score += 0.1
            else:
                score -= 0.2  # Missing cause indicates less certainty
        elif doc_type == DocumentType.METRIC:
            # Metrics with formulas are more confident
            if "formula" in insight and insight["formula"]:
                score += 0.1

        # Clamp to 0-1 range
        return max(0.0, min(1.0, score))

    def _extract_title(self, text: str, doc_type: DocumentType) -> str:
        """Extract a concise title from text."""
        # Truncate to first sentence or 80 chars
        if ". " in text:
            title = text.split(". ")[0]
        else:
            title = text[:80]
        return title.strip()

    def _infer_grain(self, dimensions: list) -> str:
        """Infer grain from dimensions."""
        if not dimensions:
            return "unknown"
        if len(dimensions) == 1:
            return f"grouped by {dimensions[0]}"
        return f"grouped by {', '.join(dimensions)}"


def generate_from_insights(insights: list) -> list:
    """
    Generate documents from a list of insights.

    Args:
        insights: List of insight dictionaries

    Returns:
        List of document dictionaries with text and metadata
    """
    generator = DocumentGenerator()
    documents = []

    for insight in insights:
        doc = generator.generate_document(insight)
        documents.append(doc)

    return documents


def extract_retrieval_metadata(text: str) -> dict:
    """
    Extract retrieval-enhancing metadata from document text without modifying the text.

    Generates:
    - queries: List of natural language questions users might ask
    - synonyms: Dict mapping key terms to alternative phrasing

    Args:
        text: Original document text (with header and content)

    Returns:
        Dictionary with keys:
        - "queries": list of query strings (may be empty)
        - "synonyms": dict mapping terms to list of synonyms (may be empty)
    """
    result = {"queries": [], "synonyms": {}}

    if not text or len(text.strip()) < 10:
        return result

    lines = text.split('\n')
    header = lines[0] if lines else ""

    # Extract first actual content line, skipping template labels
    # (e.g., skip "Definition:", "Formula:", "- Dimensions:", etc.)
    main_content = None
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if this line is a template label (ends with ":" or starts with "-")
        # and the next non-empty line exists (which would be the actual content)
        is_label = (line.endswith(':') or line.startswith('-'))
        next_idx = i + 1
        next_line = None
        while next_idx < len(lines):
            if lines[next_idx].strip():
                next_line = lines[next_idx].strip()
                break
            next_idx += 1

        if is_label and next_line:
            # Use the content after the label
            main_content = next_line
            break
        elif not is_label:
            # This line itself is content
            main_content = line
            break

        i += 1

    if not main_content:
        # Fallback: use first non-empty line
        for line in lines[1:]:
            if line.strip():
                main_content = line.strip()
                break

    if not main_content:
        return result

    # Extract key terms (capitalized words, numbers with %, $, etc.)
    key_terms = _extract_key_terms(main_content)

    # Generate query questions
    queries = _generate_queries(header, main_content, key_terms)
    if queries:
        result["queries"] = queries[:3]  # Max 3

    # Generate synonym alternatives
    synonyms = _generate_synonym_alternatives(main_content, key_terms)
    if synonyms:
        result["synonyms"] = synonyms

    return result


def enhance_for_retrieval(text: str) -> str:
    """
    Enhance document text with synonyms and query-style phrasing for better retrieval.

    Note: This function modifies the text by appending enhancement sections.
    For pipeline use, prefer extract_retrieval_metadata() and store in metadata.

    Args:
        text: Original document text (with header and content)

    Returns:
        Enhanced text with additional retrieval-friendly phrasing appended
    """
    metadata = extract_retrieval_metadata(text)

    enhancement_parts = []

    if metadata.get("queries"):
        enhancement_parts.append("\n\n# Retrieval Queries")
        enhancement_parts.append("\n" + "\n".join(f"- {q}" for q in metadata["queries"]))

    if metadata.get("synonyms"):
        enhancement_parts.append("\n\n# Alternative Terms")
        synonym_lines = []
        for term, alts in metadata["synonyms"].items():
            if alts:
                synonym_lines.append(f"- {term}: {', '.join(alts)}")
        if synonym_lines:
            enhancement_parts.append("\n" + "\n".join(synonym_lines))

    if enhancement_parts:
        return text + "".join(enhancement_parts)
    else:
        return text  # No enhancement needed


def _extract_key_terms(text: str) -> dict:
    """Extract important terms from text for synonym generation."""
    terms = {
        'dimension': None,  # e.g., category, market, region
        'metric': None,     # e.g., profit, sales, margin
        'value': None,      # e.g., highest, 14.5%, $664K
        'entities': []      # e.g., Technology, Canada
    }

    # Extract capitalized entities (proper nouns)
    import re
    words = text.split()
    for word in words:
        clean = word.strip('.,$%')
        if clean and clean[0].isupper() and len(clean) > 2:
            if clean not in terms['entities']:
                terms['entities'].append(clean)

    # Extract numbers with units
    number_pattern = r'[\d,.]+%?|\$[\d,]+K?'
    numbers = re.findall(number_pattern, text)
    if numbers:
        terms['value'] = numbers[0]

    # Extract domain keywords
    domain_terms = {
        'profit': ['profit', 'earnings', 'income', 'net income'],
        'sales': ['sales', 'revenue', 'turnover', 'income'],
        'margin': ['margin', 'profitability', 'return rate'],
        'highest': ['highest', 'maximum', 'peak', 'top', 'best'],
        'lowest': ['lowest', 'minimum', 'bottom', 'worst'],
        'increase': ['increased', 'grew', 'rose', 'climbed'],
        'decrease': ['decreased', 'fell', 'dropped', 'declined']
    }

    text_lower = text.lower()
    for category, keywords in domain_terms.items():
        for kw in keywords:
            if kw in text_lower:
                if category in ['highest', 'lowest', 'increase', 'decrease']:
                    terms['dimension'] = category
                elif category in ['profit', 'sales', 'margin']:
                    terms['metric'] = category
                break

    return terms


def _generate_queries(header: str, content: str, key_terms: dict) -> list:
    """Generate natural language query questions."""
    queries = []

    # Determine document type from header
    doc_type = None
    if '[FACT]' in header:
        doc_type = 'fact'
    elif '[TREND]' in header:
        doc_type = 'trend'
    elif '[ANOMALY]' in header:
        doc_type = 'anomaly'
    elif '[METRIC]' in header:
        doc_type = 'metric'
    elif '[QUESTION]' in header:
        doc_type = 'question'

    # Generate type-specific queries
    if doc_type == 'fact':
        dimension = key_terms.get('dimension') or 'category'
        metric = key_terms.get('metric') or 'value'
        entity = key_terms.get('entities', [''])[0] if key_terms.get('entities') else ''

        if entity:
            queries.extend([
                f"Which {dimension} has the highest {metric}?",
                f"What is the top {dimension} by {metric}?",
                f"Which {dimension} leads in {metric}?"
            ])
            if 'highest' in content.lower() or 'best' in content.lower():
                queries.append(f"Which {dimension} is best performing?")

    elif doc_type == 'trend':
        queries.extend([
            "How has performance changed over time?",
            "What is the trend direction?",
            "Is there growth or decline?"
        ])
        if 'cagr' in content.lower():
            queries.append("What is the compound annual growth rate?")

    elif doc_type == 'anomaly':
        queries.extend([
            "What is the problem area?",
            "Which category underperforms?",
            "What unexpected deviation exists?"
        ])
        if key_terms.get('entities'):
            queries.append(f"Why is {key_terms['entities'][0]} problematic?")

    elif doc_type == 'metric':
        metric = key_terms.get('metric', 'this metric')
        queries.extend([
            f"How is {metric} calculated?",
            f"What does {metric} mean?",
            f"Definition of {metric}"
        ])

    elif doc_type == 'question':
        # Already a Q&A, extract the question if possible
        if 'Q:' in content:
            q_part = content.split('Q:')[1].split('A:')[0].strip() if 'A:' in content else content
            queries.append(q_part)

    # Deduplicate and limit
    unique_queries = list(dict.fromkeys(queries))
    return unique_queries[:3]  # Max 3 queries


def _generate_synonym_alternatives(text: str, key_terms: dict) -> dict:
    """Generate synonym mappings for key terms found in text."""
    synonyms_map = {}

    # Map of term to synonyms
    synonym_dict = {
        'profit': ['earnings', 'income', 'gains', 'net income'],
        'sales': ['revenue', 'turnover', 'business income'],
        'margin': ['profitability', 'return rate', 'profit margin'],
        'highest': ['maximum', 'peak', 'top', 'greatest', 'best'],
        'lowest': ['minimum', 'bottom', 'worst'],
        'increased': ['rose', 'grew', 'climbed', 'went up'],
        'decreased': ['fell', 'dropped', 'declined', 'went down'],
        'category': ['product category', 'segment', 'class'],
        'market': ['region', 'territory', 'geography']
    }

    text_lower = text.lower()

    for term, synonyms in synonym_dict.items():
        if term in text_lower:
            # Find which synonyms are also relevant (same context)
            relevant_syns = []
            for syn in synonyms:
                if syn not in text_lower:  # Don't include if already in text
                    relevant_syns.append(syn)
            if relevant_syns:
                synonyms_map[term] = relevant_syns[:2]  # Limit to 2 synonyms

    return synonyms_map

