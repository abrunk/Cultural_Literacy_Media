"""
nb15_enrichment.py — Three-source concept enrichment.

1. Wikidata P921 (main subject) → new concept tags for films with poor extraction
2. Wikipedia Vital Articles Level 3 + Level 4 → academic importance signal
3. Quiz bowl answer frequencies (QuizDB API or QANTA fallback) → breadth signal

Fixes:
  - 12 Years a Slave: only 1 tag (Steve McQueen) → P921 adds declared subjects
  - Ben-Hur: NER extracted crew only → P921 adds ancient Rome / Biblical content
  - Spartacus: "Third Servile War" not in composite → Vital Articles / quiz bowl cover it

Outputs:
  data/p921_tags.csv               — new concept tags from Wikidata P921
  data/vital_articles.csv          — Level 3 + Level 4 vital article titles + level
  data/qb_frequencies.csv          — quiz bowl answer frequencies
  data/enriched_concepts.csv       — composite + vital + qb signals merged
  data/work_concept_tags.csv       — updated with P921 rows appended
  data/works_hirsch_scored.csv     — updated rankings

Run:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/nb15_enrichment.py
"""

import re, sys, time, json, unicodedata, warnings, io
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import requests
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache' / 'nb15'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
WIKI_API        = 'https://en.wikipedia.org/w/api.php'
QUIZDB_API      = 'https://www.quizdb.org/api/random'
QANTA_URL       = 'https://s3.amazonaws.com/my89public/quanta/qs.csv'

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'CulturalLiteracyProject/1.0 (research)',
    'Accept': 'application/json',
})
BROWSER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def clean_qb_answer(raw):
    """Normalize quiz bowl answer: strip bracketed hints, underscores, etc."""
    if not isinstance(raw, str):
        return ''
    # Remove [accept ...], [prompt on ...], etc.
    s = re.sub(r'\[.*?\]', '', raw)
    # Remove <em>...</em> HTML tags
    s = re.sub(r'<[^>]+>', '', s)
    s = s.replace('_', ' ').strip()
    return normalize(s)


def cached_get(url, cache_path, params=None, is_json=True, retry=3, delay=10):
    p = Path(cache_path)
    if p.exists():
        content = p.read_text(encoding='utf-8', errors='replace')
        if content.strip():
            return json.loads(content) if is_json else content
    for attempt in range(retry):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()
            text = resp.text
            p.write_text(text, encoding='utf-8')
            return json.loads(text) if is_json else text
        except Exception as e:
            print(f'  Attempt {attempt+1}/{retry} failed: {e}')
            if attempt < retry - 1:
                time.sleep(delay * (attempt + 1))
    return None


def run_sparql(query, retries=3):
    for attempt in range(retries):
        try:
            resp = SESSION.get(WIKIDATA_SPARQL,
                               params={'query': query, 'format': 'json'},
                               timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return [{k: v.get('value', '') for k, v in r.items()}
                    for r in data['results']['bindings']]
        except Exception as e:
            print(f'  SPARQL attempt {attempt+1}: {e}')
            if attempt < retries - 1:
                time.sleep(30 * (attempt + 1))
    return []


# ── 1. Load data ──────────────────────────────────────────────────────────────

print('\n=== 1. Load existing data ===')
corpus  = pd.read_csv(DATA_DIR / 'works_corpus.csv')
tags    = pd.read_csv(DATA_DIR / 'work_concept_tags.csv')
probs   = pd.read_csv(DATA_DIR / 'hirsch_calibrated_concepts.csv')

composite_lookup = dict(zip(probs['concept'], probs['composite_score']))
prob_lookup      = dict(zip(probs['concept'], probs['hirsch_prob']))

print(f'Corpus: {len(corpus):,}  Tags: {len(tags):,}  Concepts: {len(probs):,}')

# Low-tag films (< 5 matched concepts) — priority for P921 enrichment
matched_counts = (
    tags[tags['matched_concept'].notna()]
    .groupby('work_id').size().rename('n_matched')
)
low_tag = corpus.merge(matched_counts, on='work_id', how='left')
low_tag['n_matched'] = low_tag['n_matched'].fillna(0).astype(int)
low_tag_films = low_tag[
    (low_tag['medium'] == 'film') & (low_tag['n_matched'] < 5)
][['work_id', 'title', 'qid', 'n_matched']].dropna(subset=['qid'])

print(f'Films with <5 matched concepts: {len(low_tag_films)}')
for _, r in low_tag_films.sort_values('n_matched').head(20).iterrows():
    print(f'  [{int(r["n_matched"])}] {r["title"]}')


# ── 2. Wikidata P921 (main subjects) ─────────────────────────────────────────

print('\n=== 2. Wikidata P921 (main subjects) ===')
P921_CACHE = CACHE_DIR / 'p921_raw.json'

# Fetch P921 for ALL films with QIDs (not just low-tag — enriches everyone)
film_qids = corpus[corpus['medium'] == 'film']['qid'].dropna().tolist()
print(f'Films with QIDs: {len(film_qids)}')

if P921_CACHE.exists():
    p921_rows = json.loads(P921_CACHE.read_text(encoding='utf-8'))
    print(f'Loaded P921 cache: {len(p921_rows):,} rows')
else:
    all_rows = []
    BATCH = 60
    total = (len(film_qids) + BATCH - 1) // BATCH
    for i in range(0, len(film_qids), BATCH):
        batch = film_qids[i:i + BATCH]
        vals  = ' '.join(f'wd:{q}' for q in batch)
        query = (
            'SELECT ?item ?subject ?subjectLabel ?subjectType WHERE {\n'
            f'  VALUES ?item {{ {vals} }}\n'
            '  ?item wdt:P921 ?subject .\n'
            '  OPTIONAL { ?subject wdt:P31 ?subjectType . }\n'
            '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }\n'
            '}'
        )
        rows = run_sparql(query)
        for r in rows:
            r['qid'] = r['item'].split('/')[-1]
        all_rows.extend(rows)
        bn = i // BATCH + 1
        print(f'  Batch {bn}/{total}: {len(rows)} rows', flush=True)
        time.sleep(2)

    P921_CACHE.write_text(json.dumps(all_rows, ensure_ascii=False), encoding='utf-8')
    print(f'Saved {len(all_rows):,} rows -> {P921_CACHE}')
    p921_rows = all_rows

p921_df = pd.DataFrame(p921_rows)
if not p921_df.empty:
    p921_df['qid']   = p921_df.get('qid', p921_df.get('item', pd.Series()).str.split('/').str[-1])
    p921_df['label'] = p921_df.get('subjectLabel', pd.Series(dtype=str))
    p921_df['norm']  = p921_df['label'].apply(normalize)
    p921_df = p921_df[p921_df['norm'].str.len() >= 3].copy()
    print(f'Unique P921 subjects: {p921_df["norm"].nunique():,}')

# Build new tag rows from P921
qid_to_work = (corpus.drop_duplicates('qid')
               .set_index('qid')[['work_id', 'title', 'medium', 'year']]
               .to_dict('index'))

p921_tag_rows = []
seen_p921 = set()   # deduplicate (work_id, concept_norm)
for _, r in p921_df.iterrows():
    qid   = r.get('qid', '')
    meta  = qid_to_work.get(qid)
    if not meta:
        continue
    norm  = r['norm']
    label = r['label']
    key   = (meta['work_id'], norm)
    if key in seen_p921:
        continue
    seen_p921.add(key)
    score = composite_lookup.get(norm, np.nan)
    p921_tag_rows.append({
        'work_id'        : meta['work_id'],
        'title'          : meta['title'],
        'medium'         : meta['medium'],
        'year'           : meta['year'],
        'display_concept': label,
        'concept_norm'   : norm,
        'concept_type'   : 'theme',          # P921 = declared topic
        'relevance'      : 2,
        'matched_concept': norm if pd.notna(score) else np.nan,
        'matched_qid'    : np.nan,
        'matched_score'  : score,
        'source'         : 'p921',
    })

p921_tags = pd.DataFrame(p921_tag_rows)
if not p921_tags.empty:
    print(f'P921 tags created: {len(p921_tags):,}  '
          f'({p921_tags["matched_concept"].notna().sum():,} matched to composite)')

# Spot check for problem films
for film_name in ['12 Years a Slave', 'Ben-Hur', 'Spartacus', 'Glory']:
    sub = p921_tags[p921_tags['title'].str.contains(film_name, case=False, na=False)]
    if len(sub):
        print(f'\n  P921 for {film_name}:')
        for _, row in sub.iterrows():
            sc = f'{row["matched_score"]:.0f}' if pd.notna(row["matched_score"]) else '—'
            print(f'    {row["display_concept"]:<35}  score={sc}')

p921_tags.drop(columns=['source'], errors='ignore').to_csv(
    DATA_DIR / 'p921_tags.csv', index=False
)
print(f'\nSaved p921_tags.csv: {len(p921_tags):,} rows')


# ── 2b. Historical grounding signal ───────────────────────────────────────────

print('\n=== 2b. Historical Grounding Signal ===')

# P31 types — three tiers based on how directly educational the signal is:
#   HIST_PERSON (1.0): film's declared subject is a real person → biographical focus
#   HIST_EVENT  (0.8): specific, concrete historical event (battle, Holocaust, 9/11)
#   HIST_PERIOD (0.5): broad historical period or armed conflict used as backdrop
#   HIST_MED    (0.3): somewhat historical but abstract (ethnic group, terrorism)
HIST_PERSON = {'Q5'}          # human (real person)

HIST_EVENT = {
    'Q13418847',    # historical event (e.g. slavery in the United States)
    'Q178561',      # battle
    'Q6107280',     # revolt / insurrection
    'Q124734',      # uprising
    # Holocaust subtypes:
    'Q8461', 'Q41397', 'Q173462', 'Q459409', 'Q857833', 'Q2245405', 'Q13634374',
    # 9/11 subtypes:
    'Q217327', 'Q898712', 'Q2223653', 'Q750215',
}

HIST_PERIOD = {
    'Q11514315',    # historical period
    'Q103495',      # temporal entity / historical period variant
    'Q864113',      # armed conflict (Vietnam War, WWII as broad category)
    'Q1469686',     # Cold War (type 1)
    'Q4176199',     # Cold War (type 2)
    'Q718893',      # Pacific War
    'Q116505632',   # revolution (broad)
    'Q17544377',    # history of a place
    'Q19958368',    # culture of a country
}

HIST_MED = {
    'Q41710',       # ethnic group (e.g. Romani people, Jewish people)
    'Q466439',      # terrorism (real but abstract)
    'Q28640',       # samurai (historical social role)
}

HIST_ALPHA = 0.5  # max boost for a fully person-focused historical film (50%)

def hist_weight(qtype):
    if qtype in HIST_PERSON:
        return 1.0
    if qtype in HIST_EVENT:
        return 0.8
    if qtype in HIST_PERIOD:
        return 0.5
    if qtype in HIST_MED:
        return 0.3
    return 0.0

if not p921_df.empty:
    p921_hist = p921_df.copy()
    p921_hist['subjectType_str'] = (
        p921_hist['subjectType'].fillna('').str.split('/').str[-1]
    )
    p921_hist['hw'] = p921_hist['subjectType_str'].apply(hist_weight)

    # Map qid → work_id
    qid_wid = corpus[['qid', 'work_id']].dropna(subset=['qid'])
    p921_hist = p921_hist.merge(qid_wid, on='qid', how='left')

    # Take MAX hw per (film, subject) to handle subjects with multiple P31 types
    # (e.g. WWII has both Q11514315 and Q103495 — both HIST_HIGH, max=1.0)
    sub_hw = (
        p921_hist.dropna(subset=['work_id'])
        .groupby(['work_id', 'norm'])['hw'].max()
        .reset_index()
    )
    # Mean across all distinct subjects per film
    hist_scores = (
        sub_hw.groupby('work_id')['hw'].mean()
        .reset_index()
        .rename(columns={'hw': 'hist_ground'})
    )
else:
    hist_scores = pd.DataFrame(columns=['work_id', 'hist_ground'])

print(f'Films with P921 hist_ground data: {len(hist_scores):,}')

# Spot-check
_hg_check = {
    'Lawrence of Arabia' : 'Lawrence',
    "Schindler's List"   : 'Schindler',
    '12 Years a Slave'   : '12 Years',
    'Apocalypse Now'     : 'Apocalypse',
    'Casablanca'         : 'Casablanca',
    'Gone with the Wind' : 'Gone with',
    'Lincoln'            : 'Lincoln',
    'Zero Dark Thirty'   : 'Zero Dark',
    'E.T.'               : r'E\.T\.',
    'Raiders'            : 'Raiders',
    'The Godfather'      : '^The Godfather$',
    'Raging Bull'        : 'Raging Bull',
    'Taxi Driver'        : 'Taxi Driver',
    'Inception'          : 'Inception',
    'Forrest Gump'       : 'Forrest Gump',
}
print('\nHistorical grounding spot-check:')
for name, pat in _hg_check.items():
    r = corpus[corpus['title'].str.contains(pat, case=False, na=False, regex=True)]
    if len(r):
        wid = r.iloc[0]['work_id']
        hs  = hist_scores[hist_scores['work_id'] == wid]
        hg  = round(hs.iloc[0]['hist_ground'], 3) if len(hs) else 0.0
        print(f'  {name:<25}  hist_ground={hg:.3f}')


# ── 3. Wikipedia Vital Articles ───────────────────────────────────────────────

print('\n=== 3. Wikipedia Vital Articles ===')
VITAL_CACHE = CACHE_DIR / 'vital_articles.json'

LEVEL4_SUBPAGES = [
    'Wikipedia:Vital articles/Level/4/People',
    'Wikipedia:Vital articles/Level/4/History',
    'Wikipedia:Vital articles/Level/4/Geography',
    'Wikipedia:Vital articles/Level/4/Arts',
    'Wikipedia:Vital articles/Level/4/Philosophy and religion',
    'Wikipedia:Vital articles/Level/4/Everyday life',
    'Wikipedia:Vital articles/Level/4/Society and social sciences',
    'Wikipedia:Vital articles/Level/4/Biology and health sciences',
    'Wikipedia:Vital articles/Level/4/Physical sciences',
    'Wikipedia:Vital articles/Level/4/Technology',
    'Wikipedia:Vital articles/Level/4/Mathematics',
]

if VITAL_CACHE.exists():
    vital_data = json.loads(VITAL_CACHE.read_text(encoding='utf-8'))
    print(f'Loaded vital cache: {len(vital_data):,} articles')
else:
    vital_data = {}

    def fetch_vital_links(page_title, level):
        data = cached_get(
            WIKI_API,
            CACHE_DIR / f'vital_{level}_{re.sub(r"[^a-z0-9]", "_", page_title.lower())}.json',
            params={'action': 'parse', 'page': page_title,
                    'prop': 'links', 'format': 'json'},
        )
        if not data:
            return []
        links = data.get('parse', {}).get('links', [])
        return [lk['*'] for lk in links if lk.get('ns') == 0]

    # Level 3 (single page, ~998 articles)
    print('  Fetching Level 3...')
    l3 = fetch_vital_links('Wikipedia:Vital articles/Level/3', 3)
    for title in l3:
        vital_data[title] = 3
    print(f'    Level 3: {len(l3)} articles')
    time.sleep(1)

    # Level 4 (subpages)
    for sp in LEVEL4_SUBPAGES:
        name = sp.split('/')[-1]
        print(f'  Fetching Level 4/{name}...')
        links = fetch_vital_links(sp, f'4_{name}')
        new = 0
        for title in links:
            if title not in vital_data:
                vital_data[title] = 4
                new += 1
        print(f'    {len(links)} articles ({new} new)', flush=True)
        time.sleep(0.5)

    VITAL_CACHE.write_text(json.dumps(vital_data, ensure_ascii=False), encoding='utf-8')
    print(f'Saved vital_articles cache: {len(vital_data):,} articles')

# Build vital lookup by normalized title
vital_norm = {}
for title, level in vital_data.items():
    norm = normalize(title)
    if norm and (norm not in vital_norm or vital_norm[norm] < level):
        vital_norm[norm] = level    # keep best (lowest-numbered) level

l3_count = sum(1 for v in vital_norm.values() if v == 3)
l4_count = sum(1 for v in vital_norm.values() if v == 4)
print(f'Normalized vital concepts: {len(vital_norm):,}  (L3={l3_count}, L4={l4_count})')

# Export
vital_rows = [{'title': t, 'level': lv, 'norm': normalize(t)}
              for t, lv in vital_data.items()]
pd.DataFrame(vital_rows).to_csv(DATA_DIR / 'vital_articles.csv', index=False)
print(f'Saved vital_articles.csv')

# Spot-check classical topics
check = ['roman empire', 'julius caesar', 'slavery in the united states',
         'spartacus', 'gladiator', 'ancient rome', 'antebellum south',
         'third servile war', 'chariot racing', 'solomon northup',
         'american civil war', 'abolitionism', 'ottoman empire']
print('\nVital article spot-check:')
for c in check:
    lv = vital_norm.get(c) or vital_norm.get(c.rstrip('s')) or '—'
    print(f'  {c:<40} L{lv}')


# ── 4. Quiz Bowl frequencies (QB Reader API) ─────────────────────────────────

print('\n=== 4. Quiz Bowl answer frequencies ===')
QB_CACHE    = CACHE_DIR / 'qb_answers.json'
QBREADER_URL = 'https://www.qbreader.org/api/random-tossup'

qb_counts = {}

if QB_CACHE.exists():
    qb_counts = json.loads(QB_CACHE.read_text(encoding='utf-8'))
    print(f'Loaded QB cache: {len(qb_counts):,} distinct answers')
else:
    # QB Reader API: returns 1 tossup per request regardless of numQuestions
    # Need ~2000 requests to build a meaningful frequency distribution
    print('  Fetching from QB Reader API (random tossups)...')
    n_requests = 2000
    n_per_req  = 1
    qb_raw     = []
    errors     = 0

    for i in range(n_requests):
        cache_f = CACHE_DIR / f'qbr_{i:04d}.json'
        data = cached_get(
            QBREADER_URL,
            cache_f,
            params={'numQuestions': n_per_req},
            retry=3, delay=5,
        )
        if data and 'tossups' in data:
            qb_raw.extend(data['tossups'])
            errors = 0
        else:
            errors += 1
            if errors >= 5:
                print(f'  Too many errors after {i} requests, stopping.')
                break
        if (i + 1) % 25 == 0:
            print(f'  Batch {i+1}/{n_requests}: {len(qb_raw):,} questions so far', flush=True)
        time.sleep(0.25)

    print(f'  Collected {len(qb_raw):,} tossup questions')

    # Build frequency table — prefer answer_sanitized, fall back to answer
    _HTML_TAG = re.compile(r'<[^>]+>')
    _BRACKET  = re.compile(r'\[.*?\]|\(.*?\)')
    for item in qb_raw:
        raw = item.get('answer_sanitized') or item.get('answer', '')
        # Strip HTML and bracket hints
        raw = _HTML_TAG.sub('', str(raw))
        raw = _BRACKET.sub('', raw)
        ans = normalize(raw)
        if len(ans) >= 3:
            qb_counts[ans] = qb_counts.get(ans, 0) + 1

    QB_CACHE.write_text(json.dumps(qb_counts, ensure_ascii=False), encoding='utf-8')
    print(f'Saved QB cache: {len(qb_counts):,} distinct answers')

# Top quiz bowl answers
qb_series = pd.Series(qb_counts).sort_values(ascending=False)
print(f'\nTop 30 quiz bowl answers:')
for ans, cnt in qb_series.head(30).items():
    print(f'  [{cnt:4d}]  {ans}')

print(f'\nQuiz bowl coverage for target concepts:')
for c in ['roman empire', 'julius caesar', 'spartacus', 'slavery',
          'american civil war', 'holocaust', 'ottoman empire',
          'third servile war', 'gladiator', 'ancient rome']:
    cnt = qb_counts.get(c, 0)
    print(f'  {c:<35}  count={cnt}')

# Save frequencies CSV
qb_df = pd.DataFrame({'concept': list(qb_counts.keys()),
                      'qb_count': list(qb_counts.values())})
qb_df['qb_pctile'] = qb_df['qb_count'].rank(pct=True) * 100
qb_df.to_csv(DATA_DIR / 'qb_frequencies.csv', index=False)
print(f'Saved qb_frequencies.csv: {len(qb_df):,} concepts')

qb_pctile_lookup = dict(zip(qb_df['concept'], qb_df['qb_pctile']))


# ── 5. Build enriched concept scores ─────────────────────────────────────────

print('\n=== 5. Build enriched concept scores ===')

VITAL_SCORE = {3: 85.0, 4: 68.0}   # percentile-equivalent importance
QB_FLOOR    = 50.0                   # min qb_pctile to count as a signal
QB_WEIGHT   = 0.6                    # how much quiz bowl contributes vs vital

def effective_score(norm_concept):
    """Blended importance score: composite, vital, and quiz bowl."""
    comp     = composite_lookup.get(norm_concept, np.nan)
    vital_lv = vital_norm.get(norm_concept)
    vital_sc = VITAL_SCORE.get(vital_lv, 0.0)
    qb_pc    = qb_pctile_lookup.get(norm_concept, 0.0)
    qb_sc    = qb_pc if qb_pc >= QB_FLOOR else 0.0

    signals = [s for s in [comp, vital_sc, qb_sc] if s > 0]
    if not signals:
        return np.nan
    return max(signals)    # take the highest signal for any source


# Add enriched_score to the concepts table
probs['vital_level'] = probs['concept'].map(vital_norm)
probs['qb_pctile']   = probs['concept'].map(qb_pctile_lookup).fillna(0.0)
probs['enriched_score'] = probs['concept'].apply(effective_score)

print(f'Concepts with vital signal  : {probs["vital_level"].notna().sum():,}')
print(f'Concepts with qb signal     : {(probs["qb_pctile"] >= QB_FLOOR).sum():,}')
print(f'Concepts with enriched_score: {probs["enriched_score"].notna().sum():,}')

# Export
enriched_cols = [c for c in probs.columns
                 if c not in ['display_name'] or True]
probs.to_csv(DATA_DIR / 'enriched_concepts.csv', index=False)
print(f'Saved enriched_concepts.csv')

# Check that newly-important concepts are covered
spot = ['roman empire', 'julius caesar', 'spartacus', 'ancient rome',
        'slavery in the united states', 'american civil war', 'antebellum south',
        'holocaust', 'ottoman empire', 'chariot racing', 'gladiatorial combat',
        'third servile war', 'solomon northup', 'gladiator']
print('\nEnriched score spot-check:')
for c in spot:
    comp = composite_lookup.get(c, '—')
    vl   = vital_norm.get(c, '—')
    qb   = f'{qb_pctile_lookup.get(c, 0):.0f}'
    eff  = effective_score(c)
    eff_s = f'{eff:.0f}' if pd.notna(eff) else '—'
    print(f'  {c:<40}  comp={str(comp)[:4]:>4}  vital=L{vl}  qb={qb:>3}  → {eff_s}')


# ── 6. Merge P921 tags into work_concept_tags ─────────────────────────────────

print('\n=== 6. Merge P921 tags into work_concept_tags ===')

# Update matched_score in P921 tags using enriched scores
keep_cols = ['work_id','title','medium','year','display_concept',
             'concept_norm','concept_type','relevance',
             'matched_concept','matched_qid','matched_score']

if not p921_tags.empty:
    # Apply enriched score to P921 tags (these are new; use effective_score)
    p921_tags['matched_score']   = p921_tags['concept_norm'].apply(effective_score)
    p921_tags['matched_concept'] = p921_tags.apply(
        lambda r: r['concept_norm'] if pd.notna(r['matched_score']) else np.nan, axis=1
    )
    # Only add rows not already in the NER tag set
    existing_keys = set(zip(tags['work_id'], tags['concept_norm'].fillna('')))
    p921_new = p921_tags[
        ~p921_tags.apply(
            lambda r: (r['work_id'], r['concept_norm']) in existing_keys, axis=1
        )
    ].copy()
    print(f'New P921 tags (not already in tags): {len(p921_new):,}')

    # Keep original NER matched_scores unchanged; only add P921 rows on top
    combined = pd.concat([tags[keep_cols], p921_new[keep_cols]], ignore_index=True)
else:
    combined = tags[keep_cols].copy()

combined.to_csv(DATA_DIR / 'work_concept_tags.csv', index=False)
print(f'Saved work_concept_tags.csv: {len(combined):,} rows '
      f'({combined["matched_concept"].notna().sum():,} matched)')

# Show P921 impact on worst-covered films
for film_name in ['12 Years a Slave', 'Ben-Hur', 'Spartacus', 'Lawrence of Arabia']:
    sub = combined[combined['title'].str.contains(film_name, case=False, na=False) &
                   combined['matched_concept'].notna()]
    print(f'\n  {film_name} — now {len(sub)} matched tags:')
    for _, row in sub.sort_values('matched_score', ascending=False).head(8).iterrows():
        src = 'P921' if row.get('concept_type') == 'theme' else 'NER'
        sc  = f'{row["matched_score"]:.0f}' if pd.notna(row["matched_score"]) else '—'
        print(f'    [{row["relevance"]}|{src}] {row["display_concept"]:<35}  score={sc}')


# ── 7. Recompute enriched yields and rankings ─────────────────────────────────

print('\n=== 7. Recompute enriched yields ===')

matched_all = combined[combined['matched_concept'].notna()].copy()
matched_all['enriched_score'] = matched_all['concept_norm'].apply(effective_score).fillna(0.0)
matched_all['hirsch_prob']    = matched_all['matched_concept'].map(prob_lookup).fillna(0.0)

# Three yield measures
matched_all['hirsch_contrib']   = matched_all['relevance'] * matched_all['hirsch_prob']
matched_all['enriched_contrib'] = matched_all['relevance'] * matched_all['enriched_score'] / 100.0
matched_all['raw_contrib']      = matched_all['relevance'] * matched_all['matched_score'].fillna(0) / 100.0

yields = (
    matched_all.groupby('work_id')
    .agg(
        n_matched        = ('matched_concept', 'nunique'),
        hirsch_yield     = ('hirsch_contrib',   'sum'),
        enriched_yield   = ('enriched_contrib',  'sum'),
        raw_yield        = ('raw_contrib',        'sum'),
    )
    .reset_index()
)

corpus_clean = corpus.copy()
corpus_clean['year_num'] = pd.to_numeric(
    corpus_clean['year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
)
compare = corpus_clean.merge(yields, on='work_id', how='left')
compare = compare.merge(hist_scores, on='work_id', how='left')
compare['hist_ground'] = compare['hist_ground'].fillna(0.0)
for c in ['hirsch_yield','enriched_yield','raw_yield']:
    compare[c] = compare[c].fillna(0.0)

imp = compare['importance_score'].fillna(0)
imp_norm = 100 * imp / imp.max()

def film_ranks(yield_col):
    y = compare[compare['medium'] == 'film'][yield_col]
    y_norm = 100 * y / max(y.max(), 1e-9)
    comb   = np.sqrt(imp_norm[compare['medium'] == 'film'] * y_norm)
    return comb.rank(ascending=False, method='min').astype(int)

films_df = compare[compare['medium'] == 'film'].copy()
h_norm = 100 * films_df['hirsch_yield']   / films_df['hirsch_yield'].max()
e_norm = 100 * films_df['enriched_yield'] / films_df['enriched_yield'].max()
r_norm = 100 * films_df['raw_yield']      / films_df['raw_yield'].max()

_hist = films_df['hist_ground']
films_df['rank_hirsch']   = np.sqrt(imp_norm[films_df.index] * h_norm).rank(ascending=False, method='min').astype(int)
films_df['rank_enriched'] = (np.sqrt(imp_norm[films_df.index] * e_norm)
                              * (1 + HIST_ALPHA * _hist)).rank(ascending=False, method='min').astype(int)
films_df['rank_raw']      = np.sqrt(imp_norm[films_df.index] * r_norm).rank(ascending=False, method='min').astype(int)

print('\nImpact on target films (Hirsch → Enriched ranking):')
targets = {
    'Lawrence of Arabia' : 'Lawrence',
    "Schindler's List"   : 'Schindler',
    'Gone with the Wind' : 'Gone with',
    'Zero Dark Thirty'   : 'Zero Dark',
    '12 Years a Slave'   : '12 Years',
    'Ben-Hur'            : 'Ben-Hur',
    'Spartacus'          : 'Spartacus',
    'Casablanca'         : 'Casablanca',
    'Apocalypse Now'     : 'Apocalypse',
    'The Godfather'      : '^The Godfather$',
}
noise = {
    'Almost Famous'   : 'Almost Famous',
    'Harry Potter 1'  : 'Philosopher',
    'The Dark Knight' : 'Dark Knight',
    'The Avengers'    : '^The Avengers',
    'Interstellar'    : 'Interstellar',
}
print(f'  {"Film":<26}  Hirsch  Enriched  Δ')
print(f'  {"-"*26}  ------  --------  --')
for name, pat in {**targets, **noise}.items():
    r = films_df[films_df['title'].str.contains(pat, case=False, na=False, regex=True)]
    if len(r):
        row = r.iloc[0]
        rh, re = int(row['rank_hirsch']), int(row['rank_enriched'])
        arrow = '↑' if re < rh else ('↓' if re > rh else '=')
        delta = rh - re
        print(f'  {"✅ " if name in targets else "⚠️  "}{name:<24}  #{rh:<6}  #{re:<8}  {arrow}{abs(delta)}')


# ── 8. Top 25 films under enriched scoring ────────────────────────────────────

print('\n=== 8. Top 25 films — Enriched vs Hirsch ===')
top_enriched = films_df.sort_values('rank_enriched').head(25)
top_hirsch   = films_df.sort_values('rank_hirsch').head(25)

print(f'\n{"#":>3}  {"Enriched":50}  {"Hirsch":50}')
print('-' * 110)
enr_list = list(top_enriched['title'])
hir_list = list(top_hirsch['title'])
for i in range(25):
    et = enr_list[i][:48] if i < len(enr_list) else ''
    ht = hir_list[i][:48] if i < len(hir_list) else ''
    print(f'{i+1:3d}  {et:<50}  {ht:<50}')


# ── 9. Export updated hirsch_scored ──────────────────────────────────────────

print('\n=== 9. Export updated rankings ===')

# Recompute full combined for all works using enriched yield
all_works = compare.copy()
for c in ['hirsch_yield','enriched_yield']:
    all_works[c] = all_works[c].fillna(0.0)

hy_max = all_works['hirsch_yield'].max()
ey_max = all_works['enriched_yield'].max()
imp_max = all_works['importance_score'].max()

all_works['hyield_norm']     = 100 * all_works['hirsch_yield']   / hy_max
all_works['eyield_norm']     = 100 * all_works['enriched_yield'] / ey_max
all_works['imp_norm']        = 100 * all_works['importance_score'].fillna(0) / imp_max
all_works['hirsch_combined'] = np.sqrt(all_works['imp_norm'] * all_works['hyield_norm'])
all_works['enr_combined']    = (np.sqrt(all_works['imp_norm'] * all_works['eyield_norm'])
                                 * (1 + HIST_ALPHA * all_works['hist_ground']))

all_works = all_works.sort_values('enr_combined', ascending=False).reset_index(drop=True)
all_works['rank'] = all_works.index + 1

out_cols = ['work_id','title','medium','year_num','importance_score',
            'n_matched','hirsch_yield','enriched_yield',
            'imp_norm','hyield_norm','eyield_norm',
            'hirsch_combined','enr_combined','hist_ground','rank']
all_works[out_cols].rename(columns={'year_num': 'year'}).to_csv(
    DATA_DIR / 'works_hirsch_scored.csv', index=False
)
print(f'Saved works_hirsch_scored.csv: {len(all_works):,} works')

print('\nTOP 25 FILMS (enriched combined score):')
films_final = all_works[all_works['medium'] == 'film'].head(25)
for _, r in films_final.iterrows():
    yr = str(int(r['year_num'])) if pd.notna(r['year_num']) else '?'
    print(f"  {int(r['rank']):3d}.  {r['title'][:52]:<52}  ({yr})  "
          f"imp={r['importance_score']:.1f}  enr={r['enr_combined']:.1f}")

print('\nTOP 25 NOVELS (enriched combined score):')
novels_final = all_works[all_works['medium'] == 'novel'].head(25)
for _, r in novels_final.iterrows():
    yr = str(int(r['year_num'])) if pd.notna(r['year_num']) else '?'
    print(f"  {int(r['rank']):3d}.  {r['title'][:52]:<52}  ({yr})  "
          f"imp={r['importance_score']:.1f}  enr={r['enr_combined']:.1f}")

print('\nDone.')
