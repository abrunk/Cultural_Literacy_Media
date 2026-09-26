"""
nb12_discover.py — standalone discovery loop for Notebook 12.

Runs the Wikipedia category-based candidate discovery without nbconvert's
timeout. All results are cached in data/cache/nb12/ so re-runs are instant
for already-processed seeds.

Run with:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/nb12_discover.py
"""

import pandas as pd
import re
import json
import time
import unicodedata
import requests
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache' / 'nb12'
for sub in ('search', 'cats', 'members'):
    (CACHE_DIR / sub).mkdir(parents=True, exist_ok=True)

WIKI_API  = 'https://en.wikipedia.org/w/api.php'
HEADERS   = {'User-Agent': 'CulturalLiteracyProject/1.0 (research)'}
SLEEP_SEC = 0.15

SCORE_THRESH        = 75
SEED_TYPES          = {'PERSON', 'GPE', 'EVENT', 'NORP', 'LOC', 'WORK_OF_ART', 'ORG'}
MAX_CATS_PER_PREFIX = 2
MAX_CATS_PER_CONCEPT= 4
MAX_MEMBERS_PER_CAT = 100
MAX_PREFIXES_PER_SEED = 3

FILM_RE   = re.compile(r'\bfilm\b', re.I)
NOVEL_RE  = re.compile(r'\b(novel|short story collection|story collection|novella)\b', re.I)
BOOK_RE   = re.compile(r'\bbook\b', re.I)
REJECT_RE = re.compile(
    r'\b(documentary|television|TV series|miniseries|mini-series|short film|'
    r'animated series|web series|video game|album|song|play|opera|musical|'
    r'non-fiction|nonfiction|reference book|autobiography|biography|memoir|'
    r'essay|podcast|radio|manga|comic book|graphic novel|picture book)\b', re.I
)
YEAR_RE = re.compile(r'\b(1[89]\d\d|20[012]\d)\b')


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


# ── API helpers ─────────────────────────────────────────────────────────────

def _wiki_get(params, retries=3):
    params.setdefault('format', 'json')
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {}


def search_wiki(query, limit=1):
    data = _wiki_get({'action': 'query', 'list': 'search',
                      'srsearch': query, 'srlimit': limit})
    try:
        return data['query']['search'][0]['title']
    except (KeyError, IndexError):
        return None


def find_media_categories(prefix, limit=5):
    data = _wiki_get({'action': 'query', 'list': 'allcategories',
                      'acprefix': prefix, 'aclimit': limit})
    cats = [c['*'] for c in data.get('query', {}).get('allcategories', [])]
    media_kw = {'film', 'novel', 'fiction', 'book', 'adapt', 'literature',
                'cinema', 'screen', 'story', 'stories'}
    return [c for c in cats if any(k in c.lower() for k in media_kw)]


def fetch_category_members(cat_title, limit=100):
    data = _wiki_get({'action': 'query', 'list': 'categorymembers',
                      'cmtitle': f'Category:{cat_title}',
                      'cmlimit': limit, 'cmnamespace': 0})
    return [m['title'] for m in data.get('query', {}).get('categorymembers', [])]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _safe_key(s, maxlen=80):
    return re.sub(r'[^\w\s-]', '_', str(s).lower())[:maxlen]

def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None

def _save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

def cached_search(display_name):
    path = CACHE_DIR / 'search' / f'{_safe_key(display_name)}.json'
    cached = _load(path)
    if cached is not None:
        return cached.get('canonical')
    canonical = search_wiki(display_name)
    _save(path, {'display_name': display_name, 'canonical': canonical})
    time.sleep(SLEEP_SEC)
    return canonical

def cached_find_cats(prefix):
    path = CACHE_DIR / 'cats' / f'{_safe_key(prefix)}.json'
    cached = _load(path)
    if cached is not None:
        return cached.get('cats', [])
    cats = find_media_categories(prefix)
    _save(path, {'prefix': prefix, 'cats': cats})
    time.sleep(SLEEP_SEC)
    return cats

def cached_fetch_members(cat_name):
    path = CACHE_DIR / 'members' / f'{_safe_key(cat_name)}.json'
    cached = _load(path)
    if cached is not None:
        return cached.get('members', [])
    members = fetch_category_members(cat_name)
    _save(path, {'cat': cat_name, 'members': members})
    time.sleep(SLEEP_SEC)
    return members


# ── Prefix generation ────────────────────────────────────────────────────────

def get_prefixes(canonical, entity_type):
    c = canonical
    if entity_type == 'PERSON':
        return [f'Films about {c}', f'Books about {c}', f'{c} in fiction']
    elif entity_type == 'WORK_OF_ART':
        return [f'Films based on {c}', f'Adaptations of {c}', f'Novels based on {c}']
    elif entity_type == 'EVENT':
        return [f'{c} films', f'{c} fiction', f'Films about {c}', f'{c} in fiction']
    elif entity_type == 'GPE':
        return [f'Films set in {c}', f'Novels set in {c}', f'Films set in the {c}']
    elif entity_type == 'LOC':
        return [f'Films set in {c}', f'{c} in fiction', f'Novels set in {c}']
    elif entity_type == 'NORP':
        return [f'Films about {c}', f'{c} films', f'{c}-language films']
    elif entity_type == 'ORG':
        return [f'Films about {c}', f'Novels about {c}', f'Films set in {c}']
    else:
        return [f'Films about {c}']


# ── Main discovery loop ──────────────────────────────────────────────────────

def main():
    composite = pd.read_csv(DATA_DIR / 'cultural_salience_composite_v2.csv',
                            dtype={'qid': str}, low_memory=False)
    gaps   = pd.read_csv(DATA_DIR / 'concept_gaps.csv')
    corpus = pd.read_csv(DATA_DIR / 'works_corpus.csv')

    gap_norms    = set(gaps['concept'].str.lower().dropna())
    corpus_norm  = set(corpus['norm_title'].dropna())

    seeds = composite[
        (composite['composite_score'] >= SCORE_THRESH) &
        (composite['entity_type'].isin(SEED_TYPES)) &
        (composite['display_name'].notna()) &
        (composite['display_name'].str.split().str.len() >= 2) &
        (~composite['display_name'].str.match(r'^Q[0-9]+')) &
        (composite['display_name'].str.len() >= 4)
    ].copy().drop_duplicates('display_name').sort_values(
        'composite_score', ascending=False
    ).reset_index(drop=True)

    seeds['is_gap'] = seeds['concept'].str.lower().isin(gap_norms)

    n_search_cached = len(list((CACHE_DIR / 'search').glob('*.json')))
    print(f'Seeds: {len(seeds):,}  (already cached: {n_search_cached:,} search files)')
    print(f'Gap seeds: {seeds["is_gap"].sum():,}  Covered: {(~seeds["is_gap"]).sum():,}')
    print()

    candidate_map = defaultdict(lambda: {
        'total_score': 0.0, 'gap_score': 0.0,
        'all_concepts': [], 'gap_concepts': [],
    })

    _REJECT_TITLE = re.compile(
        r'^(List of|Category:|Template:|Wikipedia:|Portal:)|\(disambiguation\)$', re.I
    )

    t0     = time.time()
    n_hits = 0
    n_new  = 0  # API calls that weren't cached

    for i, (_, seed) in enumerate(seeds.iterrows()):
        display     = seed['display_name']
        entity_type = seed['entity_type']
        score       = float(seed['composite_score'])
        is_gap      = bool(seed['is_gap'])

        search_path = CACHE_DIR / 'search' / f'{_safe_key(display)}.json'
        was_cached  = search_path.exists()

        canonical = cached_search(display)
        if not was_cached:
            n_new += 1
        if not canonical:
            continue

        prefixes   = get_prefixes(canonical, entity_type)[:MAX_PREFIXES_PER_SEED]
        found_cats = []
        for prefix in prefixes:
            cats = cached_find_cats(prefix)
            for cat in cats[:MAX_CATS_PER_PREFIX]:
                if cat not in found_cats:
                    found_cats.append(cat)
        found_cats = found_cats[:MAX_CATS_PER_CONCEPT]

        if not found_cats:
            continue
        n_hits += 1

        for cat in found_cats:
            members = cached_fetch_members(cat)
            for title in members:
                if _REJECT_TITLE.search(title):
                    continue
                cand = candidate_map[title]
                cand['total_score']  += score
                cand['all_concepts'].append(display)
                if is_gap:
                    cand['gap_score']  += score
                    cand['gap_concepts'].append(display)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / max(elapsed, 1)
            eta     = (len(seeds) - i - 1) / max(rate, 0.001)
            print(f'[{i+1:5d}/{len(seeds)}]  {elapsed:6.0f}s  rate={rate:.1f}/s  '
                  f'ETA~{eta/60:.0f}m  new_calls={n_new}  candidates={len(candidate_map):,}',
                  flush=True)

    elapsed = time.time() - t0
    print(f'\nDiscovery complete in {elapsed:.0f}s ({elapsed/60:.1f}m)')
    print(f'  Seeds with categories: {n_hits:,} / {len(seeds):,}')
    print(f'  Unique candidates    : {len(candidate_map):,}')
    print(f'  New API calls        : {n_new:,}')

    # Save candidate map for NB12 to load
    out = {title: info for title, info in candidate_map.items()}
    out_path = DATA_DIR / 'cache' / 'nb12' / 'candidate_map.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print(f'\nSaved candidate_map ({len(out):,} entries) -> {out_path}')
    print('Run NB12 cells 6-8 (validation + export) to finish.')


if __name__ == '__main__':
    main()
