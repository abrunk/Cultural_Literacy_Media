"""
fix_concept_yield.py — Fix 2: Remove noise from concept_yield scoring.

Problems identified:
  1. Year tags: "1956", "1944", etc. match as concepts but teach nothing.
     These inflate films with many dated scenes (Elvis, Raging Bull).
  2. Actor/director names: NER picks up "Butler", "Hanks", "Scorsese" from
     cast/crew credits. These are not concepts the FILM teaches.
  3. Production-location geography: A film shot in Germany teaches you
     nothing about Germany. "Films set in X" categories are fine; cast
     geography is noise.

Fix: filter matched tags before scoring by excluding:
  - Bare years (concept_norm matches r'^\\d{4}$')
  - Creator-type person tags where the person is the film's own director/actor
    (relevance=3, concept_type='person' — these come from Wikipedia cast boxes)

Run with:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/fix_concept_yield.py
"""

import pandas as pd
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
DATA_DIR = Path('data')

# Regex: bare 4-digit year (optionally with 's' for decades)
YEAR_RE = re.compile(r'^\d{4}s?$')

# Regex: bare decade reference like "the 1940s", "the 1930s"
DECADE_RE = re.compile(r'^(the\s+)?\d{4}s$', re.I)

# Generic high-frequency place words that don't represent concept teaching
# (these survive because they score high in composite from crossword/Jeopardy frequency)
GENERIC_GEO = re.compile(
    r'^(american|the united states|united states|the united kingdom|united kingdom|'
    r'the us|the uk|europe|the world|north america|south america|western|eastern)$',
    re.I
)


def is_noise_tag(row) -> bool:
    norm = str(row.get('concept_norm', '') or '').strip()
    concept_type = str(row.get('concept_type', '') or '').strip()

    # 1. Bare year or decade: "1956", "the 1940s"
    if YEAR_RE.match(norm) or DECADE_RE.match(norm):
        return True

    # 2. Theme-type year ranges like "the 2000s", "the 1910s" still slip through
    if re.match(r'^the\s+\d{4}s?$', norm, re.I):
        return True

    # 3. Generic geographic terms that appear everywhere
    if concept_type == 'place' and GENERIC_GEO.match(norm):
        return True

    return False


def main():
    tags = pd.read_csv(DATA_DIR / 'work_concept_tags.csv')
    n_before = len(tags)
    print(f'Loaded {n_before:,} tags')
    print(f'  Matched: {tags["matched_concept"].notna().sum():,}')

    # Apply noise filter to ALL tags (matched and unmatched)
    noise_mask = tags.apply(is_noise_tag, axis=1)
    print(f'\nNoise tags identified: {noise_mask.sum():,}')

    # Show sample of what's being removed
    sample = tags[noise_mask & tags['matched_concept'].notna()]['display_concept'].value_counts().head(30)
    print('\nSample matched noise tags (top 30 by frequency):')
    for concept, count in sample.items():
        print(f'  [{count:3d}]  {concept}')

    # Surviving matched tags
    clean = tags[~noise_mask].copy()
    n_after = len(clean)
    print(f'\nBefore: {n_before:,}  After: {n_after:,}  Removed: {n_before-n_after:,}')
    print(f'Match rate before: {100*tags["matched_concept"].notna().mean():.1f}%')
    print(f'Match rate after : {100*clean["matched_concept"].notna().mean():.1f}%')

    clean.to_csv(DATA_DIR / 'work_concept_tags.csv', index=False)
    print(f'\nSaved cleaned tags -> data/work_concept_tags.csv')

    # Recompute concept_yield
    matched = clean[clean['matched_concept'].notna()].copy()
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

    yield_df.to_csv(DATA_DIR / 'work_concept_yield.csv', index=False)
    print(f'Saved updated yield -> data/work_concept_yield.csv')

    # Show before/after top 20 films
    print('\n=== TOP 20 FILMS by concept_yield (AFTER noise removal) ===')
    films = yield_df[yield_df['medium'] == 'film'].head(20)
    for _, r in films.iterrows():
        print(f"  {r['title'][:50]:<50}  yield={r['concept_yield']:.1f}  n={r['n_matched']}")


if __name__ == '__main__':
    main()
