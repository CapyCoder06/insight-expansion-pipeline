"""
Template definitions for document generation.
"""

from .types import DocumentType

# Document templates
TEMPLATES = {
    DocumentType.FACT: """# [FACT] {title}

{description}

- Dimensions: {dimensions}
- Metrics: {metrics}
- Grain: {grain}
""",
    DocumentType.TREND: """# [TREND] {title}

{description}

- Dimension: {dimension}
- Metrics: {metrics}
- Grain: {grain}
- Trend: {trend}
""",
    DocumentType.ANOMALY: """# [ANOMALY] {title}

{description}

- Dimension: {dimension}
- Metric: {metric}
- Issue: {issue}
- Possible Cause: {possible_cause}
""",
    DocumentType.METRIC: """# [METRIC] {metric}

Definition:
{definition}

Formula:
{formula}

Interpretation:
{interpretation}
""",
    DocumentType.QUESTION: """# [QUESTION]

Q: {question}

A: {answer}
""",
}
