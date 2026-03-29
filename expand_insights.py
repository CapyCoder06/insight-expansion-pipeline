#!/usr/bin/env python
"""
CLI script to expand insights from the EDA dataset.

Generates a large set of structured insights from the data, with optional
input seed file (for reporting only) and configurable target count.
"""

import argparse
import json
import sys
import csv
from statistics import mean
from pathlib import Path

try:
    from pipeline.config import load_config
    config_available = True
except ImportError:
    config_available = False

# Import Deduplicator for signature-based deduplication
try:
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from insight_expansion.deduplicator import Deduplicator
    DEDUPLICATOR_AVAILABLE = True
except ImportError:
    DEDUPLICATOR_AVAILABLE = False


def format_currency(value):
    return f"${value:,.0f}"

def format_percent(value):
    return f"{value:.1f}%"

def generate_insights_from_data(csv_path: str, target_count: int = 100):
    """Generate insights strictly following:
    - type_hint in {'fact','anomaly','trend','metric','question'} (we use only fact and anomaly)
    - dimensions list contains exactly ONE dimension (category or region)
    - All values grounded in dataset.
    """
    # Load CSV data
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'region': row['region'],
                'category': row['category'],
                'revenue': float(row['revenue']),
                'profit': float(row['profit']),
                'margin': float(row['margin'])
            })

    # Build aggregates
    cat_agg = {}
    region_agg = {}
    for row in data:
        cat = row['category']
        reg = row['region']
        # init
        if cat not in cat_agg:
            cat_agg[cat] = {'revenue': 0, 'profit': 0, 'count': 0}
        if reg not in region_agg:
            region_agg[reg] = {'revenue': 0, 'profit': 0, 'count': 0}
        # accumulate
        cat_agg[cat]['revenue'] += row['revenue']
        cat_agg[cat]['profit'] += row['profit']
        cat_agg[cat]['count'] += 1
        region_agg[reg]['revenue'] += row['revenue']
        region_agg[reg]['profit'] += row['profit']
        region_agg[reg]['count'] += 1

    # Compute margins
    for agg in [cat_agg, region_agg]:
        for ent, vals in agg.items():
            vals['margin'] = (vals['profit'] / vals['revenue'] * 100) if vals['revenue'] > 0 else 0.0

    insights = []

    def add_insight(text, dimension, metrics, type_hint='fact', **extra):
        """Append an insight ensuring one dimension and valid type_hint."""
        # allowed types: fact, anomaly, trend, metric, question
        if type_hint not in ['fact', 'anomaly', 'trend', 'metric', 'question']:
            type_hint = 'fact'  # default safe mapping
        # Ensure dimension is a list with exactly one element
        if not isinstance(dimension, list) or len(dimension) != 1:
            raise ValueError(f"Insight must have single dimension, got: {dimension}")
        ins = {
            'text': text,
            'dimensions': dimension,
            'metrics': metrics,
            'type_hint': type_hint
        }
        ins.update(extra)
        insights.append(ins)

    # Helper: generate facts per entity set (categories or regions)
    def generate_entity_insights(agg_dict, dim_name):
        entities = list(agg_dict.keys())
        n_entities = len(entities)
        # Precompute rankings and averages for each metric
        metrics = ['revenue', 'profit', 'margin']
        rankings = {}
        averages = {}
        for metric in metrics:
            # sort descending by value
            sorted_entities = sorted(entities, key=lambda e: agg_dict[e][metric], reverse=True)
            rankings[metric] = sorted_entities
            avg_val = sum(agg_dict[e][metric] for e in entities) / n_entities if n_entities else 0
            averages[metric] = avg_val

        # For each entity and metric, create base fact and possibly qualifier fact
        for metric in metrics:
            for ent in entities:
                val = agg_dict[ent][metric]
                val_str = format_percent(val) if metric == 'margin' else format_currency(val)
                rank = rankings[metric].index(ent) + 1

                # Base fact: "X metric is value"
                base_text = f"{ent} {metric} is {val_str}"
                add_insight(base_text, [dim_name], [metric], 'fact')

                # Qualifier fact: with highest/lowest/above avg/below avg
                qualifier = None
                if rank == 1:
                    qualifier = "highest"
                elif rank == n_entities:
                    qualifier = "lowest"
                else:
                    avg = averages[metric]
                    if val > avg * 1.1:
                        qualifier = "above average"
                    elif val < avg * 0.9:
                        qualifier = "below average"
                if qualifier:
                    qual_text = f"{ent} has {qualifier} {metric} of {val_str}"
                    add_insight(qual_text, [dim_name], [metric], 'fact')

        # Add moderate margin facts (10-15%) for all entities
        for ent in entities:
            marg = agg_dict[ent]['margin']
            if 10 <= marg < 15:
                m_str = format_percent(marg)
                add_insight(f"{ent} has moderate margin: {m_str}", [dim_name], ['margin'], 'fact')

    # Generate for categories
    generate_entity_insights(cat_agg, 'category')
    # Generate for regions
    generate_entity_insights(region_agg, 'region')

    # Add anomaly insights for low margin (<3%) and negative profit at entity level
    for ent, agg in cat_agg.items():
        if agg['margin'] < 3:
            m_str = format_percent(agg['margin'])
            rev_str = format_currency(agg['revenue'])
            add_insight(
                f"{ent} has critically low margin: {m_str} with {rev_str} revenue",
                ['category'],
                ['margin', 'revenue'],
                'anomaly',
                issue='Margin below 3%',
                possible_cause='High costs or low pricing'
            )
    for ent, agg in region_agg.items():
        if agg['margin'] < 3:
            m_str = format_percent(agg['margin'])
            rev_str = format_currency(agg['revenue'])
            add_insight(
                f"{ent} has critically low margin: {m_str} with {rev_str} revenue",
                ['region'],
                ['margin', 'revenue'],
                'anomaly',
                issue='Margin below 3%',
                possible_cause='High costs or low pricing'
            )

    # Add combined profit+margin facts (still single dimension)
    for ent, agg in cat_agg.items():
        p_str = format_currency(agg['profit'])
        m_str = format_percent(agg['margin'])
        add_insight(f"{ent}: profit {p_str}, margin {m_str}", ['category'], ['profit','margin'], 'fact')
    for ent, agg in region_agg.items():
        p_str = format_currency(agg['profit'])
        m_str = format_percent(agg['margin'])
        add_insight(f"{ent}: profit {p_str}, margin {m_str}", ['region'], ['profit','margin'], 'fact')

    # Add high revenue facts (limited number) to help reach target
    # These are not essential but add variety
    for cat in cat_agg:
        if cat_agg[cat]['revenue'] > 300000:
            rev_str = format_currency(cat_agg[cat]['revenue'])
            add_insight(f"{cat} has high revenue: {rev_str}", ['category'], ['revenue'], 'fact')
    for reg in region_agg:
        if region_agg[reg]['revenue'] > 200000:
            rev_str = format_currency(region_agg[reg]['revenue'])
            add_insight(f"{reg} has high revenue: {rev_str}", ['region'], ['revenue'], 'fact')

    # Apply intelligent deduplication using Deduplicator (signature + text similarity)
    if DEDUPLICATOR_AVAILABLE:
        deduped_insights = Deduplicator.deduplicate(insights)
    else:
        # Fallback: simple text-based deduplication
        seen_texts = set()
        deduped_insights = []
        for ins in insights:
            if ins['text'] not in seen_texts:
                seen_texts.add(ins['text'])
                deduped_insights.append(ins)

    # Truncate to target count if needed
    if len(deduped_insights) > target_count:
        final = deduped_insights[:target_count]
    else:
        final = deduped_insights

    # Sort for consistency
    final.sort(key=lambda x: (x['type_hint'], x['text']))
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Expand insights from EDA dataset into a larger set of structured insights."
    )
    parser.add_argument(
        '--input',
        type=str,
        default='insights_sample.json',
        help='Path to seed insights JSON file (default: insights_sample.json). Used for reporting only.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='insights_expanded.json',
        help='Path to output JSON file (default: insights_expanded.json)'
    )
    parser.add_argument(
        '--target',
        type=int,
        default=None,
        help='Target number of insights to generate (default from config or 100)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/eda_structured.csv',
        help='Path to the EDA CSV data file (default: data/eda_structured.csv)'
    )

    args = parser.parse_args()

    try:
        # Load config if available
        cfg = None
        if config_available:
            try:
                cfg = load_config()
            except Exception as e:
                print(f"Warning: Could not load config: {e}. Using defaults.", file=sys.stderr)

        # Determine target count: CLI > config > default 100
        target = args.target
        if target is None:
            target = cfg.get('expansion.target_count', 100) if cfg else 100

        # Determine output file: CLI > config > default
        output_path = args.output
        if output_path == 'insights_expanded.json':  # default from argparse
            output_path = cfg.get('expansion.output_file', 'output/insights_expanded.json') if cfg else 'output/insights_expanded.json'

        # Get dedup_threshold from config (for future semantic deduplication)
        dedup_threshold = cfg.get('expansion.dedup_threshold', 0.85) if cfg else 0.85

        # Check that input file exists (required)
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            return 1

        # Count seed insights (for reporting; not used for generation)
        seed_count = 0
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                seed_data = json.load(f)
            seeds = seed_data if isinstance(seed_data, list) else seed_data.get('insights', [])
            seed_count = len(seeds)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file {args.input}: {e}", file=sys.stderr)
            return 1

        # Check CSV exists
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV data file not found: {args.csv}", file=sys.stderr)
            return 1

        print(f"Generating insights from CSV: {csv_path}")
        print(f"Target count: {target}")

        # Generate insights
        insights = generate_insights_from_data(csv_path=str(csv_path), target_count=target)

        # Prepare output with metadata
        output = {
            "insights": insights,
            "metadata": {
                "total_generated": len(insights),
                "source": str(csv_path),
                "seed_count": seed_count,
                "target_count": target,
                "dedup_threshold": dedup_threshold
            }
        }

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Failed to write output file {args.output}: {e}", file=sys.stderr)
            return 1

        print(f"\nSuccessfully generated {len(insights)} insights")
        print(f"  Seeds in: {seed_count}")
        print(f"  Insights generated: {len(insights)}")
        print(f"  Final count: {len(insights)}")
        print(f"  Output written to: {output_path}")

        if len(insights) < target:
            print(f"  Note: Generated count is below target ({target}).", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
