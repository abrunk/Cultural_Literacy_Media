"""
nb12_validate.py — Validation + export for NB12 candidate discovery.

Loads candidate_map.json saved by nb12_discover.py, batch-validates each
candidate title via Wikipedia wikibase-shortdesc to confirm it is a film
or novel, deduplicates against the existing corpus, and exports
data/candidate_works.csv.

Run:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/nb12_validate.py
"""

import json
import re
import sys
import time
import unicodedata

import pandas as pd
import requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache' / 'nb12'
SHORTDESC_CACHE = CACHE_DIR / 'shortdescs.json'

WIKI_API  = 'https://en.wikipedia.org/w/api.php'
HEADERS   = {'User-Agent': 'CulturalLiteracyProject/1.0 (research)'}
SLEEP_SEC = 0.15

FILM_RE   = re.compile(r'\bfilm\b', re.I)
NOVEL_RE  = re.compile(r'\b(novel|short story collection|story collection|novella)\b', re.I)
BOOK_RE   = re.compile(r'\bbook\b', re.I)
REJECT_RE = re.compile(
    r'\b(documentary|television|TV series|miniseries|mini-series|short film|'
    r'animated series|web series|video game|album|song|play|opera|musical|'
    r'non-fiction|nonfiction|reference book|autobiography|biography|memoir|'
    r'essay|podcast|radio|manga|comic book|graphic novel|picture book)\b', re.I
)
YEAR_RE        = re.compile(r'\b(1[89]\d\d|20[012]\d)\b')
_YEAR_SUFFIX   = re.compile(r'\s*\((\d{4})\s*[^)]*\)\s*$')
_MEDIUM_SUFFIX = re.compile(r'\s*\((film|novel|book|movie)\)\s*$', re.I)
_REJECT_TITLE  = re.compile(
    r'^(List of|Category:|Template:|Wikipedia:|Portal:)|\(disambiguation\)$', re.I
)


def normalize(text):
    if not isinstance(text, str):
        return ''
    s = unicodedata.normalize('NFKD', text)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r'^(the|a|an)\s+', '', s)
    s = re.sub(r"[,;:!?'\"()\[\]]", '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def classify_medium(shortdesc):
    if not shortdesc or REJECT_RE.search(shortdesc):
        return None
    if FILM_RE.search(shortdesc):
        return 'film'
    if NOVEL_RE.search(shortdesc):
        return 'novel'
    if BOOK_RE.search(shortdesc) and not FILM_RE.search(shortdesc):
        return 'novel'
    return None


def batch_fetch_shortdesc(titles, existing_cache):
    """Fetch shortdescs in batches of 50; uses per-run cache dict to avoid re-fetching."""
    result = dict(existing_cache)
    need = [t for t in titles if t not in result]
    print(f'  {len(result):,} already cached, {len(need):,} to fetch')
    total_batches = (len(need) + 49) // 50
    for bi, i in enumerate(range(0, len(need), 50)):
        batch = need[i:i + 50]
        try:
            params = {
                'action': 'query', 'format': 'json',
                'titles': '|'.join(batch),
                'prop': 'pageprops', 'ppprop': 'wikibase-shortdesc',
            }
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            for page in data.get('query', {}).get('pages', {}).values():
                result[page.get('title', '')] = page.get('pageprops', {}).get('wikibase-shortdesc', '')
        except Exception as e:
            print(f'  Batch {bi+1} error: {e}')
        if (bi + 1) % 50 == 0 or bi + 1 == total_batches:
            print(f'  Batch {bi+1}/{total_batches} done', flush=True)
        time.sleep(SLEEP_SEC)
    return result


# ── 1. Load candidate_map ────────────────────────────────────────────────────

map_path = CACHE_DIR / 'candidate_map.json'
print(f'Loading candidate_map from {map_path}...')
with open(map_path, encoding='utf-8') as f:
    candidate_map = json.load(f)
print(f'  {len(candidate_map):,} candidate titles')

# ── 2. Load corpus for dedup ─────────────────────────────────────────────────

corpus = pd.read_csv(DATA_DIR / 'works_corpus.csv')
corpus_norm = set(corpus['norm_title'].dropna())
print(f'Corpus: {len(corpus):,} works  ({len(corpus_norm):,} normalized titles)')

# ── 3. Batch shortdesc validation ────────────────────────────────────────────

# Load existing shortdesc cache if present
existing_sd = {}
if SHORTDESC_CACHE.exists():
    try:
        existing_sd = json.loads(SHORTDESC_CACHE.read_text(encoding='utf-8'))
        print(f'Shortdesc cache: {len(existing_sd):,} entries')
    except Exception:
        pass

all_titles = [t for t in candidate_map if not _REJECT_TITLE.search(t)]
print(f'\nBatch-validating {len(all_titles):,} titles...')
t0 = time.time()
desc_map = batch_fetch_shortdesc(all_titles, existing_sd)
elapsed = time.time() - t0
print(f'Done in {elapsed:.0f}s')

# Save updated shortdesc cache
SHORTDESC_CACHE.write_text(
    json.dumps(desc_map, ensure_ascii=False), encoding='utf-8'
)
print(f'Saved shortdesc cache: {len(desc_map):,} entries')

# ── 4. Build candidates DataFrame ────────────────────────────────────────────

rows = []
for title, info in candidate_map.items():
    if _REJECT_TITLE.search(title):
        continue
    desc   = desc_map.get(title, '')
    medium = classify_medium(desc)
    if medium is None:
        continue

    year_m = _YEAR_SUFFIX.search(title)
    year   = int(year_m.group(1)) if year_m else None
    if year is None:
        ym2  = YEAR_RE.search(desc)
        year = int(ym2.group(1)) if ym2 else None

    clean = _YEAR_SUFFIX.sub('', title).strip()
    clean = _MEDIUM_SUFFIX.sub('', clean).strip()
    norm  = normalize(clean)

    if norm in corpus_norm:
        continue

    all_c = list(dict.fromkeys(info['all_concepts']))
    gap_c = list(dict.fromkeys(info['gap_concepts']))

    rows.append({
        'wiki_title'         : title,
        'clean_title'        : clean,
        'norm_title'         : norm,
        'medium'             : medium,
        'year'               : year,
        'shortdesc'          : desc,
        'total_concept_score': round(info['total_score'], 2),
        'gap_fill_score'     : round(info['gap_score'], 2),
        'n_total_concepts'   : len(all_c),
        'n_gap_concepts'     : len(gap_c),
        'top_concept'        : all_c[0] if all_c else '',
        'all_concepts'       : '|'.join(all_c),
        'gap_concepts'       : '|'.join(gap_c),
    })

candidates = pd.DataFrame(rows).sort_values(
    'total_concept_score', ascending=False
).reset_index(drop=True)

print(f'\nCandidates after validation + corpus dedup:')
print(f'  Total  : {len(candidates):,}')
print(f'  Films  : {(candidates["medium"]=="film").sum():,}')
print(f'  Novels : {(candidates["medium"]=="novel").sum():,}')

# ── 5. Analysis ───────────────────────────────────────────────────────────────

cols = ['wiki_title', 'year', 'total_concept_score', 'gap_fill_score',
        'n_total_concepts', 'n_gap_concepts', 'top_concept']

print('\nTOP 40 FILMS by total_concept_score:')
top_films = candidates[candidates['medium'] == 'film'].head(40)
print(top_films[cols].to_string(index=False, float_format='{:.1f}'.format))

print('\nTOP 40 FILMS by gap_fill_score:')
top_gap = (candidates[candidates['medium'] == 'film']
           .sort_values('gap_fill_score', ascending=False).head(40))
print(top_gap[cols].to_string(index=False, float_format='{:.1f}'.format))

print('\nTOP 40 NOVELS by total_concept_score:')
top_novels = candidates[candidates['medium'] == 'novel'].head(40)
print(top_novels[cols].to_string(index=False, float_format='{:.1f}'.format))

# Score distribution
p90 = candidates['total_concept_score'].quantile(0.9)
top = candidates[candidates['total_concept_score'] >= p90]
print(f'\ntotal_concept_score distribution:')
print(candidates['total_concept_score'].describe(
    percentiles=[.5, .75, .9, .95, .99]
).to_string())
print(f'\nTop 10% (score>={p90:.0f}): {len(top):,} works  '
      f'({(top["medium"]=="film").sum()} films, {(top["medium"]=="novel").sum()} novels)')

# ── 6. Export ─────────────────────────────────────────────────────────────────

out_path = DATA_DIR / 'candidate_works.csv'
candidates.to_csv(out_path, index=False)
print(f'\nSaved {len(candidates):,} candidates -> {out_path}')
print('Done.')
