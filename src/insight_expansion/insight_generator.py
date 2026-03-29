"""Insight generator using pattern and data."""

from typing import Dict, List
from .pattern_extractor import PatternExtractor

# Allowed type_hint values
ALLOWED_TYPES = {"fact", "trend", "anomaly", "metric", "question"}


class InsightGenerator:
    """Generates new insights by applying patterns to real data."""

    @staticmethod
    def generate_insights(seed: Dict, query_engine) -> List[Dict]:
        """Generate insights from a seed by expanding across dimensions and entities."""
        # Normalize type_hint: map legacy/invalid values to standard types
        seed = seed.copy()  # Avoid mutating caller
        seed_type = seed.get("type_hint", "")
        # Explicit mapping for known legacy types
        if seed_type in ("comparison", "opportunity"):
            seed["type_hint"] = "fact"
        elif seed_type not in ALLOWED_TYPES:
            seed["type_hint"] = "fact"

        pattern = PatternExtractor.extract(seed, query_engine)
        primary_dim = pattern["primary_dimension"]
        alternate_dims = pattern.get("alternate_dimensions", [])
        dimensions_to_expand = [primary_dim] + alternate_dims
        # Ensure only non-empty dimensions are expanded (avoid invalid empty dimension)
        dimensions_to_expand = [d for d in dimensions_to_expand if d]

        generated = []
        for current_dim in dimensions_to_expand:
            # Determine which entities to process based on type
            if seed.get("type_hint") == "anomaly":
                # Use outlier detection to find relevant entities
                metric = pattern["metrics"][0] if pattern["metrics"] else None
                if metric == "profit":
                    entities = query_engine.detect_outliers(current_dim, metric, "negative")
                elif metric == "margin":
                    entities = query_engine.detect_outliers(current_dim, metric, "below_threshold", threshold=3.0)
                else:
                    entities = query_engine.get_unique_values(current_dim)  # fallback
            else:
                entities = query_engine.get_unique_values(current_dim)

            for entity in entities:
                # Decide whether to split metrics
                if InsightGenerator._should_split_metrics(pattern["metrics"], seed["type_hint"]):
                    for metric in pattern["metrics"]:
                        insight = InsightGenerator._generate_single_metric_insight(seed, pattern, entity, metric, current_dim, query_engine)
                        if insight:
                            generated.append(insight)
                else:
                    insight = InsightGenerator._generate_combined_insight(seed, pattern, entity, current_dim, query_engine)
                    if insight:
                        generated.append(insight)
        # Final validation: ensure every insight has exactly one dimension
        for insight in generated:
            if len(insight["dimensions"]) != 1:
                raise ValueError(f"Insight must have exactly one dimension, got: {insight['dimensions']}")
        return generated

    @staticmethod
    def _should_split_metrics(metrics: List[str], type_hint: str) -> bool:
        if len(metrics) <= 1:
            return False
        if type_hint in ["fact", "anomaly"]:
            return True
        return False

    @staticmethod
    def _generate_combined_insight(seed, pattern, entity, current_dim, query_engine):
        metrics_data = query_engine.get_entity_metrics(current_dim, entity, pattern["metrics"])
        if not metrics_data:
            return None

        qualifier = InsightGenerator.compute_qualifier(pattern, entity, metrics_data, query_engine, current_dim)

        # Use first metric for template (others are included in metrics list but not in text)
        metric = pattern["metrics"][0]
        value = metrics_data.get(metric, 0)
        value_str = InsightGenerator._format_value(value, metric)

        # Build text
        text = pattern["text_format"].format(
            entity=entity,
            qualifier=qualifier,
            metric=metric,
            value=value_str,
            verb="has"
        )

        new_insight = {
            "text": text,
            "dimensions": [current_dim],
            "metrics": pattern["metrics"],
            "type_hint": seed["type_hint"],
        }

        # Ensure exactly one dimension
        if len(new_insight["dimensions"]) != 1:
            raise ValueError(f"Insight must have exactly one dimension, got: {new_insight['dimensions']}")

        # Copy optional fields only if still valid
        if pattern.get("optional_fields") and InsightGenerator._optional_fields_still_valid(pattern["optional_fields"], entity, pattern):
            new_insight.update(pattern["optional_fields"])

        return new_insight

    @staticmethod
    def _generate_single_metric_insight(seed, pattern, entity, metric, current_dim, query_engine):
        metrics_data = query_engine.get_entity_metrics(current_dim, entity, [metric])
        if not metrics_data:
            return None

        qualifier = InsightGenerator.compute_qualifier(pattern, entity, metrics_data, query_engine, current_dim)
        value = metrics_data.get(metric, 0)
        value_str = InsightGenerator._format_value(value, metric)

        text = pattern["text_format"].format(
            entity=entity,
            qualifier=qualifier,
            metric=metric,
            value=value_str,
            verb="has"
        )

        new_insight = {
            "text": text,
            "dimensions": [current_dim],
            "metrics": [metric],
            "type_hint": seed["type_hint"],
        }

        # Ensure exactly one dimension
        if len(new_insight["dimensions"]) != 1:
            raise ValueError(f"Insight must have exactly one dimension, got: {new_insight['dimensions']}")

        if pattern.get("optional_fields") and InsightGenerator._optional_fields_still_valid(pattern["optional_fields"], entity, pattern):
            new_insight.update(pattern["optional_fields"])

        return new_insight

    @staticmethod
    def compute_qualifier(pattern: Dict, entity: str, metrics_data: Dict, query_engine, current_dim: str) -> str:
        comp_type = pattern["comparison_type"]
        metric = pattern["metrics"][0] if pattern["metrics"] else None

        if comp_type == "rank":
            total_entities = len(query_engine.get_unique_values(current_dim))
            rank = query_engine.get_entity_rank(current_dim, metric, entity)
            value = metrics_data.get(metric, 0)
            avg = query_engine.get_average(current_dim, metric)

            if rank == 1:
                return "highest"
            elif rank == total_entities:
                return "lowest"
            elif value > avg * 1.1:
                return "above average"
            elif value < avg * 0.9:
                return "below average"
            else:
                return "around average"

        elif comp_type == "outlier":
            value = metrics_data.get(metric, 0)
            if metric == "profit":
                if value < 0:
                    return "negative"
            elif metric == "margin":
                if value < 3.0:
                    return "below threshold"
            return "outlier"  # default label

        else:
            return ""

    @staticmethod
    def _optional_fields_still_valid(optional_fields: Dict, entity: str, pattern: Dict) -> bool:
        return entity == pattern.get("original_entity", "")

    @staticmethod
    def _format_value(value: float, metric: str) -> str:
        if metric == "margin":
            return f"{value:.1f}%"
        else:
            return f"${value:,.0f}"

    @staticmethod
    def generate_rank_insights(query_engine) -> List[Dict]:
        """
        Generate insights about entity ranking by revenue.
        Produces insights with a distinct metric 'revenue_rank' to add diversity.
        Returns a list of insight dicts.
        """
        insights = []
        # Get available dimensions
        dimensions = query_engine.get_available_dimensions()
        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            # Compute ranking by revenue
            ranked = query_engine.get_ranking(dim, 'revenue')  # list of (entity, revenue)
            # Build a map entity -> rank (1-indexed)
            rank_map = {}
            for idx, (ent, _) in enumerate(ranked):
                rank_map[ent] = idx + 1
            # Create insight for each entity
            for ent in entities:
                rank = rank_map.get(ent, len(entities) + 1)  # if not found, assign high rank
                text = f"{ent} revenue rank is {rank}"
                insight = {
                    "text": text,
                    "dimensions": [dim],
                    "metrics": ["revenue_rank"],
                    "type_hint": "fact"
                }
                insights.append(insight)
        return insights

    @staticmethod
    def generate_comparison_insights(query_engine) -> List[Dict]:
        """Generate insights comparing top and bottom entities for each metric."""
        insights = []
        dimensions = query_engine.get_available_dimensions()
        metrics = ["revenue", "profit", "margin"]

        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            if len(entities) < 2:
                continue
            for metric in metrics:
                ranked = query_engine.get_ranking(dim, metric)
                if len(ranked) < 2:
                    continue
                top_entity, top_val = ranked[0]
                bottom_entity, bottom_val = ranked[-1]
                ratio = top_val / bottom_val if bottom_val != 0 else float('inf')

                if metric == "margin":
                    top_str = f"{top_val:.1f}%"
                    bottom_str = f"{bottom_val:.1f}%"
                else:
                    top_str = f"${top_val:,.0f}"
                    bottom_str = f"${bottom_val:,.0f}"

                text = (f"Among all {dim}s, {top_entity} has the highest {metric} ({top_str}), "
                        f"while {bottom_entity} has the lowest ({bottom_str}), a {ratio:.1f}x difference.")
                insight = {
                    "text": text,
                    "dimensions": [dim],
                    "metrics": [f"{metric}_comparison"],
                    "type_hint": "fact",
                }
                insights.append(insight)
        return insights

    @staticmethod
    def generate_ratio_insights(query_engine) -> List[Dict]:
        """Generate ratio insights between pairs of categories (entity A vs B)."""
        insights = []
        # Only categories (3 entities) to limit combinatorial explosion
        dim = "category"
        entities = query_engine.get_unique_values(dim)
        if len(entities) < 2:
            return insights
        metrics = ["revenue", "profit", "margin"]
        # Build values for each entity
        values = {}
        for ent in entities:
            vals = query_engine.get_entity_metrics(dim, ent, metrics)
            values[ent] = vals

        # For each metric, generate all unordered pairs where first > second
        for metric in metrics:
            # Sort entities by metric value descending
            sorted_entities = sorted(entities, key=lambda e: values[e].get(metric, 0), reverse=True)
            for i in range(len(sorted_entities)):
                for j in range(i + 1, len(sorted_entities)):
                    A = sorted_entities[i]
                    B = sorted_entities[j]
                    val_A = values[A].get(metric, 0)
                    val_B = values[B].get(metric, 0)
                    if val_B == 0:
                        continue
                    ratio = val_A / val_B
                    if ratio <= 1.0:
                        continue  # only greater
                    if metric == "margin":
                        valA_str = f"{val_A:.1f}%"
                        valB_str = f"{val_B:.1f}%"
                    else:
                        valA_str = f"${val_A:,.0f}"
                        valB_str = f"${val_B:,.0f}"
                    text = f"{A} {metric} is {ratio:.1f}x higher than {B} ({valA_str} vs {valB_str})."
                    insight = {
                        "text": text,
                        "dimensions": [dim],
                        "metrics": [f"{metric}_ratio"],
                        "type_hint": "fact",
                    }
                    insights.append(insight)
        return insights

    @staticmethod
    def generate_aggregate_insights(query_engine) -> List[Dict]:
        """Generate insights about top N share of total."""
        insights = []
        dimensions = query_engine.get_available_dimensions()
        metrics = ["revenue", "profit", "margin"]
        N = 3

        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            if len(entities) < N:
                continue
            for metric in metrics:
                # Compute total across all entities
                total = 0.0
                ent_vals = {}
                for ent in entities:
                    val = query_engine.get_entity_metrics(dim, ent, [metric]).get(metric, 0)
                    ent_vals[ent] = val
                    total += val
                if total == 0:
                    continue
                # Sort descending and sum top N
                sorted_ents = sorted(entities, key=lambda e: ent_vals[e], reverse=True)
                topN_sum = sum(ent_vals[e] for e in sorted_ents[:N])
                share = (topN_sum / total) * 100
                unit = "%" if metric == "margin" else ""
                # For margin, share is a bit unusual; but okay.
                text = f"Top {N} {dim}s by {metric} account for {share:.1f}% of total {metric}."
                insight = {
                    "text": text,
                    "dimensions": [dim],
                    "metrics": [f"{metric}_top{N}_share"],
                    "type_hint": "fact",
                }
                insights.append(insight)
        return insights

    @staticmethod
    def generate_gap_insights(query_engine) -> List[Dict]:
        """Generate insights about gap (max - min) for each metric."""
        insights = []
        dimensions = query_engine.get_available_dimensions()
        metrics = ["revenue", "profit", "margin"]

        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            if len(entities) < 2:
                continue
            for metric in metrics:
                ranked = query_engine.get_ranking(dim, metric)
                if len(ranked) < 2:
                    continue
                max_val = ranked[0][1]
                min_val = ranked[-1][1]
                gap = max_val - min_val
                if metric == "margin":
                    gap_str = f"{gap:.1f} percentage points"
                else:
                    gap_str = f"${gap:,.0f}"
                text = f"Gap between highest and lowest {metric} across {dim}s is {gap_str}."
                insight = {
                    "text": text,
                    "dimensions": [dim],
                    "metrics": [f"{metric}_gap"],
                    "type_hint": "fact",
                }
                insights.append(insight)
        return insights

    @staticmethod
    def generate_threshold_insights(query_engine) -> List[Dict]:
        """Generate threshold-related insights: average and count below average."""
        insights = []
        dimensions = query_engine.get_available_dimensions()
        metrics = ["revenue", "profit", "margin"]

        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            for metric in metrics:
                # Compute average and count
                count = 0
                total_val = 0.0
                for ent in entities:
                    val = query_engine.get_entity_metrics(dim, ent, [metric]).get(metric, 0)
                    total_val += val
                avg = total_val / len(entities) if entities else 0.0
                # Count below average
                for ent in entities:
                    val = query_engine.get_entity_metrics(dim, ent, [metric]).get(metric, 0)
                    if val < avg:
                        count += 1
                # Insight for average
                if metric == "margin":
                    avg_str = f"{avg:.1f}%"
                else:
                    avg_str = f"${avg:,.0f}"
                text_avg = f"Average {metric} across all {dim}s is {avg_str}."
                insight_avg = {
                    "text": text_avg,
                    "dimensions": [dim],
                    "metrics": [f"{metric}_average"],
                    "type_hint": "fact",
                }
                insights.append(insight_avg)

                # Insight for count below average
                total_ents = len(entities)
                text_count = f"{count} out of {total_ents} {dim}s have {metric} below the average of {avg_str}."
                insight_count = {
                    "text": text_count,
                    "dimensions": [dim],
                    "metrics": [f"{metric}_below_avg_count"],
                    "type_hint": "fact",
                }
                insights.append(insight_count)
        return insights

    @staticmethod
    def generate_anomaly_variations_insights(query_engine) -> List[Dict]:
        """Generate anomaly count insights (e.g., count of entities with negative profit or low margin)."""
        insights = []
        dimensions = query_engine.get_available_dimensions()
        # Low margin threshold
        margin_threshold = 3.0
        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            # Count negative profit
            neg_profit_count = 0
            low_margin_count = 0
            for ent in entities:
                metrics = query_engine.get_entity_metrics(dim, ent, ["profit", "margin"])
                profit = metrics.get("profit", 0)
                margin = metrics.get("margin", 0)
                if profit < 0:
                    neg_profit_count += 1
                if margin < margin_threshold:
                    low_margin_count += 1
            # Insights for counts
            total = len(entities)
            if neg_profit_count > 0:
                text_neg = (f"{neg_profit_count} out of {total} {dim}s have negative profit.")
                insight_neg = {
                    "text": text_neg,
                    "dimensions": [dim],
                    "metrics": ["negative_profit_count"],
                    "type_hint": "anomaly",
                }
                insights.append(insight_neg)
            if low_margin_count > 0:
                text_low = (f"{low_margin_count} out of {total} {dim}s have margin below {margin_threshold}%.")
                insight_low = {
                    "text": text_low,
                    "dimensions": [dim],
                    "metrics": ["low_margin_count"],
                    "type_hint": "anomaly",
                }
                insights.append(insight_low)
        return insights

    @staticmethod
    def generate_trend_insights(query_engine) -> List[Dict]:
        """Generate trend insights: leader ratio compared to average."""
        insights = []
        dimensions = query_engine.get_available_dimensions()
        metrics = ["revenue", "profit", "margin"]

        for dim in dimensions:
            entities = query_engine.get_unique_values(dim)
            if len(entities) < 2:
                continue
            for metric in metrics:
                # Compute average
                total = 0.0
                ent_vals = {}
                for ent in entities:
                    val = query_engine.get_entity_metrics(dim, ent, [metric]).get(metric, 0)
                    ent_vals[ent] = val
                    total += val
                avg = total / len(entities) if entities else 0.0
                if avg == 0:
                    continue
                # Find top entity
                top_entity = max(entities, key=lambda e: ent_vals[e])
                top_val = ent_vals[top_entity]
                ratio = top_val / avg
                if metric == "margin":
                    val_str = f"{top_val:.1f}%"
                    avg_str = f"{avg:.1f}%"
                else:
                    val_str = f"${top_val:,.0f}"
                    avg_str = f"${avg:,.0f}"
                text = f"{top_entity} leads all {dim}s in {metric}, {ratio:.1f}x the average ({avg_str})."
                insight = {
                    "text": text,
                    "dimensions": [dim],
                    "metrics": [f"{metric}_leader_ratio"],
                    "type_hint": "trend",
                }
                insights.append(insight)
        return insights
