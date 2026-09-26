"""
clean_tags.py — Fix 1: remove Wikipedia administrative category noise from
work_concept_tags.csv.

The category parser in Notebook 10 extracts concepts from Wikipedia categories
like "Films about World War II" → "World War II" (good), but also passes through
categories that didn't match any strip pattern:
  "Best Picture Academy Award winners"
  "1952 drama films"
  "BAFTA winners"
  etc.

These 'theme' type tags are administrative/metadata labels, not cultural concepts.
This script removes them (matched tags are always kept regardless of type).

Run with:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  PYTHONPATH=.venv/Lib/site-packages python scripts/clean_tags.py
"""

import pandas as pd
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path('data')
TAGS_PATH = DATA_DIR / 'work_concept_tags.csv'

# Patterns that indicate a 'theme' concept is an administrative Wikipedia
# category, not a genuine cultural concept.
ADMIN_RE = re.compile(r"""
    \b(films?|movies?|novels?|books?)\b      # medium descriptors still in string
  | \baward\b | \bwinners?\b | \bwinning\b    # industry awards
  | \bprize\b | \bnominated\b | \blaureate
  | \bpictures\b                             # studio names ("Universal Pictures")
  | \bstudio\b | \bdistrib\w+\b
  | \bbafta\b | \boscar\b | \bgrammy\b
  | \bemmy\b | \bsaturn\b
  | \badapted\b | \bdirected.by\b            # production metadata
  | \bwritten.by\b | \bproduced.by\b
  | \bshot\s+in\b | \bfilmed\s+in\b
  | \bscreenplay\b | \bcinematograph\w*\b
  | \bsoundtrack\b | \bscore\b
  | \bimax\b | \btechnicolor\b | \bwidescreen\b
  | \bblack.and.white\b
  | \bpeople\b | \bbirths?\b | \bdeaths?\b   # Wikipedia admin categories
  | \bregistry\b | \bcatalog\w*\b
  | \bsequels?\b | \bfranchise\b
  | ^\d{4}\s                                 # year-prefixed entries ("1952 drama")
  | \b\d{4}s?\s+in\b                         # "1980s in film"
  | \blanguage\b                             # "English-language films"
  | \bindependent\b | \bproduction\b
  | \bwarner\s+bros\b | \buniversal\b
  | \bcolumbia\b | \bparamount\b
  | \b20th.century.fox\b | \bdisney\b
  | \bnetflix\b | \bhbo\b
""", re.VERBOSE | re.IGNORECASE)


def is_admin_garbage(concept_norm: str) -> bool:
    return bool(ADMIN_RE.search(concept_norm))


def main():
    tags = pd.read_csv(TAGS_PATH)
    n_before = len(tags)
    print(f"Loaded {n_before:,} tags from {TAGS_PATH}")

    # Always keep tags that already matched the composite dataset
    matched_mask = tags['matched_concept'].notna()
    print(f"  Already matched (always kept): {matched_mask.sum():,}")

    # For unmatched theme tags: apply the garbage filter
    is_theme_unmatched = (tags['concept_type'] == 'theme') & (~matched_mask)
    print(f"  Unmatched theme tags (candidates for removal): {is_theme_unmatched.sum():,}")

    garbage_mask = is_theme_unmatched & tags['concept_norm'].apply(is_admin_garbage)
    print(f"  Identified as admin garbage: {garbage_mask.sum():,}")

    # Spot-check: show a sample of what's being removed
    sample_removed = tags[garbage_mask]['display_concept'].value_counts().head(20)
    print(f"\nSample removed (top 20 by frequency):")
    for concept, count in sample_removed.items():
        print(f"  [{count:3d}]  {concept}")

    # Spot-check: show a sample of unmatched themes that survive
    surviving_themes = tags[is_theme_unmatched & ~garbage_mask]
    print(f"\nSurviving unmatched themes ({len(surviving_themes):,}) — sample:")
    for concept in surviving_themes['display_concept'].value_counts().head(20).index:
        print(f"  {concept}")

    # Apply filter
    clean_tags = tags[~garbage_mask].copy()
    n_after = len(clean_tags)
    print(f"\nBefore: {n_before:,} tags")
    print(f"After : {n_after:,} tags  (removed {n_before - n_after:,})")
    print(f"New match rate: {100 * clean_tags['matched_concept'].notna().mean():.1f}%  "
          f"(was {100 * tags['matched_concept'].notna().mean():.1f}%)")

    # Overwrite the CSV
    clean_tags.to_csv(TAGS_PATH, index=False)
    print(f"\nSaved cleaned tags → {TAGS_PATH}")

    # Also recompute yield
    matched = clean_tags[clean_tags['matched_concept'].notna()].copy()
    matched['weighted'] = matched['relevance'] * matched['matched_score']
    yield_df = (
        matched.groupby('work_id')
        .agg(
            n_concepts=('concept_norm', 'nunique'),
            n_matched=('matched_concept', 'nunique'),
            concept_yield=('weighted', 'sum'),
            avg_score=('matched_score', 'mean'),
        )
        .reset_index()
    )

    works = pd.read_csv(DATA_DIR / 'works_corpus.csv')
    yield_df = yield_df.merge(
        works[['work_id', 'title', 'medium', 'year', 'importance_score']],
        on='work_id', how='left'
    ).sort_values('concept_yield', ascending=False)

    yield_path = DATA_DIR / 'work_concept_yield.csv'
    yield_df.to_csv(yield_path, index=False)
    print(f"Saved updated yield → {yield_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
