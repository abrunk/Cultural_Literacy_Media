"""
nb14_prestige.py — Prestige signal enrichment + importance_score rebuild.

Fetches:
  - Wikidata P444: Metacritic and Rotten Tomatoes scores
  - Wikidata P166/P1411: Oscar wins/noms, BAFTA wins/noms
  - TSPDT top-1000 film ranking (They Shoot Pictures Don't They)
  - Letterboxd Top 250 (or IMDB mirror fallback)

Rebuilds importance_score with age-adjusted weights:
  - Pre-2010:  35% prestige lists + 25% awards + 25% critics + 15% sitelinks
  - 2010+:     20% prestige lists + 30% awards + 35% critics + 15% sitelinks

Outputs:
  data/prestige_signals.csv
  data/works_corpus.csv         (updated importance_score + new signal cols)
  data/works_hirsch_scored.csv  (final film/novel rankings)

Run:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/nb14_prestige.py
"""

import re, time, io, sys, unicodedata, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from rapidfuzz import fuzz, process

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache' / 'nb14'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'CulturalLiteracyProject/1.0 (research)',
    'Accept': 'application/json',
})
BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}
BATCH = 80


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_sparql(query, retries=3, delay=30):
    for attempt in range(retries):
        try:
            resp = SESSION.get(WIKIDATA_SPARQL,
                               params={'query': query, 'format': 'json'},
                               timeout=120)
            resp.raise_for_status()
            data = resp.json()
            rows = [{k: v.get('value', '') for k, v in r.items()}
                    for r in data['results']['bindings']]
            return pd.DataFrame(rows)
        except Exception as e:
            print(f'  SPARQL attempt {attempt+1} failed: {e}')
            if attempt < retries - 1:
                print(f'  Retrying in {delay}s...')
                time.sleep(delay)
    return pd.DataFrame()


def pctile(series):
    return series.rank(pct=True, na_option='keep') * 100


def normalize_title(text):
    if not isinstance(text, str):
        return ''
    s = unicodedata.normalize('NFKD', text)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r'^(the|a|an)\s+', '', s)
    s = re.sub(r"[,;:!?'\"()\[\]]", '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def parse_score_string(s):
    if not isinstance(s, str):
        return np.nan
    s = s.strip()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*/\s*(\d+)', s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den > 0:
            val = round(100.0 * num / den, 1)
            return val if val <= 100 else np.nan
    m = re.match(r'^(\d+(?:\.\d+)?)\s*%', s)
    if m:
        return min(float(m.group(1)), 100.0)
    m = re.match(r'^(\d+)$', s)
    if m:
        v = float(m.group(1))
        return v if v <= 100 else np.nan
    return np.nan


def match_list_to_corpus(list_df, corpus_df,
                          title_col='title_norm', year_col='year',
                          rank_col='tspdt_rank', threshold=82):
    corpus_df = corpus_df.copy()
    corpus_df['_norm'] = corpus_df['norm_title'].fillna('')
    corpus_df['_year'] = pd.to_numeric(corpus_df['year'], errors='coerce').fillna(0).astype(int)
    corpus_norms = corpus_df['_norm'].tolist()
    matched, unmatched = {}, []

    for _, row in list_df.iterrows():
        norm = str(row.get(title_col, ''))
        list_year = row.get(year_col)
        list_year = int(list_year) if pd.notna(list_year) else None
        rank = row[rank_col]

        exact = corpus_df[corpus_df['_norm'] == norm]
        if len(exact) == 1:
            matched[exact.iloc[0]['work_id']] = rank
            continue
        if len(exact) > 1 and list_year:
            yr = exact[exact['_year'] == list_year]
            if len(yr) >= 1:
                matched[yr.iloc[0]['work_id']] = rank
                continue
            matched[exact.iloc[0]['work_id']] = rank
            continue

        result = process.extractOne(norm, corpus_norms, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= threshold:
            cands = corpus_df[corpus_df['_norm'] == result[0]]
            if list_year:
                yr_match = cands[(cands['_year'] - list_year).abs() <= 3]
                if len(yr_match) >= 1:
                    matched[yr_match.iloc[0]['work_id']] = rank
                    continue
            if len(cands) >= 1:
                matched[cands.iloc[0]['work_id']] = rank
                continue

        unmatched.append({'title': row.get('title_col', norm), 'year': list_year, 'rank': rank})

    print(f'  Matched: {len(matched):,}  Unmatched: {len(unmatched):,}')
    if unmatched[:3]:
        print('  Sample unmatched:', unmatched[:3])
    return matched


# ── 1. Load corpus ────────────────────────────────────────────────────────────

print('\n=== 1. Load corpus ===')
corpus = pd.read_csv(DATA_DIR / 'works_corpus.csv')
print(f'Corpus: {len(corpus):,} works')
print(corpus['medium'].value_counts().to_string())

qid_works = corpus[corpus['qid'].notna()].copy()
all_qids  = qid_works['qid'].tolist()
print(f'QIDs available: {len(all_qids):,}')


# ── 2. Wikidata review scores (P444) ─────────────────────────────────────────

print('\n=== 2. Wikidata review scores (P444) ===')
REVIEW_CACHE = CACHE_DIR / 'sparql_review_scores.csv'

if REVIEW_CACHE.exists():
    review_raw = pd.read_csv(REVIEW_CACHE)
    print(f'Loaded cache: {len(review_raw):,} rows')
else:
    all_rows = []
    total_batches = (len(all_qids) + BATCH - 1) // BATCH
    for i in range(0, len(all_qids), BATCH):
        batch = all_qids[i:i + BATCH]
        values_str = ' '.join(f'wd:{q}' for q in batch)
        query = (
            'SELECT ?item ?score ?reviewer ?reviewerLabel WHERE {\n'
            f'  VALUES ?item {{ {values_str} }}\n'
            '  ?item p:P444 ?stmt .\n'
            '  ?stmt ps:P444 ?score .\n'
            '  OPTIONAL { ?stmt pq:P447 ?reviewer . }\n'
            '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }\n'
            '}'
        )
        df = run_sparql(query)
        if len(df):
            df['qid'] = df['item'].str.extract(r'(Q\d+)$')
            all_rows.append(df)
        batch_num = i // BATCH + 1
        print(f'  Batch {batch_num}/{total_batches}: {len(df)} rows', flush=True)
        time.sleep(1.5)

    review_raw = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    review_raw.to_csv(REVIEW_CACHE, index=False)
    print(f'Saved {len(review_raw):,} rows -> {REVIEW_CACHE}')

if not review_raw.empty and 'reviewerLabel' in review_raw.columns:
    print('Reviewer labels:')
    print(review_raw['reviewerLabel'].value_counts().head(8).to_string())

# Parse Metacritic and RT
mc_scores = pd.DataFrame(columns=['qid', 'metacritic'])
rt_scores = pd.DataFrame(columns=['qid', 'rt_score'])

if not review_raw.empty and 'reviewerLabel' in review_raw.columns:
    mc_mask = (
        review_raw['reviewerLabel'].str.contains('Metacritic', case=False, na=False) |
        review_raw.get('reviewer', pd.Series()).str.contains('Q150248', na=False)
    )
    mc_rows = review_raw[mc_mask].copy()
    mc_rows['metacritic'] = mc_rows['score'].apply(parse_score_string)
    mc_scores = (mc_rows.dropna(subset=['metacritic'])
                 .groupby('qid')['metacritic'].max().reset_index())

    rt_mask = (
        review_raw['reviewerLabel'].str.contains('Rotten Tomatoes', case=False, na=False) |
        review_raw.get('reviewer', pd.Series()).str.contains('Q105584', na=False)
    )
    rt_rows = review_raw[rt_mask].copy()
    rt_rows['rt_score'] = rt_rows['score'].apply(parse_score_string)
    rt_scores = (rt_rows.dropna(subset=['rt_score'])
                 .groupby('qid')['rt_score'].max().reset_index())

print(f'Metacritic scores: {len(mc_scores):,}  RT scores: {len(rt_scores):,}')


# ── 3. Wikidata awards (P166 wins, P1411 noms) ────────────────────────────────

print('\n=== 3. Wikidata awards ===')

def fetch_awards(property_id, cache_file, label):
    cache_path = CACHE_DIR / cache_file
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f'Loaded {label} cache: {len(df):,} rows')
        return df
    all_rows = []
    total_batches = (len(all_qids) + BATCH - 1) // BATCH
    for i in range(0, len(all_qids), BATCH):
        batch = all_qids[i:i + BATCH]
        values_str = ' '.join(f'wd:{q}' for q in batch)
        query = (
            'SELECT ?item ?award ?awardLabel WHERE {\n'
            f'  VALUES ?item {{ {values_str} }}\n'
            f'  ?item p:{property_id} ?stmt .\n'
            f'  ?stmt ps:{property_id} ?award .\n'
            '  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }\n'
            '}'
        )
        df = run_sparql(query)
        if len(df):
            df['qid'] = df['item'].str.extract(r'(Q\d+)$')
            all_rows.append(df)
        batch_num = i // BATCH + 1
        print(f'  [{label}] Batch {batch_num}/{total_batches}: {len(df)} rows', flush=True)
        time.sleep(1.5)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    result.to_csv(cache_path, index=False)
    print(f'Saved {len(result):,} rows -> {cache_path}')
    return result

wins_raw = fetch_awards('P166', 'sparql_wins.csv', 'wins')
noms_raw = fetch_awards('P1411', 'sparql_noms.csv', 'noms')


def aggregate_awards(df, col_prefix):
    empty = pd.DataFrame(columns=['qid', f'{col_prefix}_oscar', f'{col_prefix}_bafta'])
    if df.empty or 'awardLabel' not in df.columns:
        return empty
    oscar_mask = df['awardLabel'].str.contains('Academy Award', case=False, na=False)
    bafta_mask = df['awardLabel'].str.contains('BAFTA', case=False, na=False)
    oscar_c = (df[oscar_mask].groupby('qid').size()
               .clip(upper=20).reset_index(name=f'{col_prefix}_oscar'))
    bafta_c = (df[bafta_mask].groupby('qid').size()
               .clip(upper=20).reset_index(name=f'{col_prefix}_bafta'))
    return oscar_c.merge(bafta_c, on='qid', how='outer')

wins_agg = aggregate_awards(wins_raw, 'wins')
noms_agg = aggregate_awards(noms_raw, 'noms')

print(f'Oscar wins : {wins_agg["wins_oscar"].notna().sum() if "wins_oscar" in wins_agg.columns else 0} works')
print(f'Oscar noms : {noms_agg["noms_oscar"].notna().sum() if "noms_oscar" in noms_agg.columns else 0} works')
print(f'BAFTA wins : {wins_agg["wins_bafta"].notna().sum() if "wins_bafta" in wins_agg.columns else 0} works')

# Spot check
spot = corpus[corpus['title'].str.contains(
    'Lawrence|Schindler|Martian|Elvis|Zero Dark|Casablanca|Godfather',
    case=False, na=False
)][['title', 'qid']].copy()
spot = spot.merge(wins_agg, on='qid', how='left').merge(noms_agg, on='qid', how='left')
print('\nAward spot check:')
print(spot.fillna(0).to_string(index=False))


# ── 4. TSPDT scrape ───────────────────────────────────────────────────────────

print('\n=== 4. They Shoot Pictures (TSPDT) ===')
TSPDT_CACHE = CACHE_DIR / 'tspdt_raw.html'
TSPDT_URL   = 'https://www.theyshootpictures.com/gf1000_all1000films_table.php'

if TSPDT_CACHE.exists():
    html_text = TSPDT_CACHE.read_text(encoding='utf-8', errors='replace')
    print(f'Loaded TSPDT cache ({len(html_text):,} chars)')
else:
    print('Fetching TSPDT...')
    resp = requests.get(TSPDT_URL, headers=BROWSER_HEADERS, timeout=60)
    resp.raise_for_status()
    html_text = resp.text
    TSPDT_CACHE.write_text(html_text, encoding='utf-8')
    print(f'Fetched {len(html_text):,} chars')

tables = pd.read_html(io.StringIO(html_text))
print(f'Tables found: {len(tables)}')
tspdt_raw = tables[0].copy()
print(f'Shape: {tspdt_raw.shape}  Columns: {list(tspdt_raw.columns)}')

tspdt = tspdt_raw.dropna(how='all').copy()
# Positional column renaming: Pos | YearRank | Title | Director | Year | Country | Mins
if len(tspdt.columns) == 7:
    tspdt.columns = ['pos', 'year_rank', 'title', 'director', 'year', 'country', 'mins']
elif len(tspdt.columns) == 6:
    tspdt.columns = ['pos', 'title', 'director', 'year', 'country', 'mins']
else:
    print(f'WARNING: {len(tspdt.columns)} columns, using col0=pos, col2=title, col3=dir, col4=year')
    tspdt = tspdt.rename(columns={tspdt.columns[0]: 'pos',
                                    tspdt.columns[2]: 'title',
                                    tspdt.columns[3]: 'director',
                                    tspdt.columns[4]: 'year'})

tspdt['tspdt_rank'] = pd.to_numeric(tspdt['pos'], errors='coerce')
tspdt['year']       = pd.to_numeric(tspdt.get('year', pd.Series()), errors='coerce')
tspdt = tspdt.dropna(subset=['tspdt_rank'])
tspdt = tspdt[tspdt['tspdt_rank'].between(1, 1000)].reset_index(drop=True)
tspdt['title_norm'] = tspdt['title'].apply(normalize_title)

print(f'TSPDT: {len(tspdt):,} films')
print(tspdt[['tspdt_rank', 'title', 'year']].head(10).to_string(index=False))

print('\nMatching TSPDT to corpus...')
tspdt_map = match_list_to_corpus(tspdt, corpus,
                                  title_col='title_norm',
                                  year_col='year',
                                  rank_col='tspdt_rank')
print(f'TSPDT in corpus: {len(tspdt_map):,} works')


# ── 5. Letterboxd Top 250 (or IMDB fallback) ─────────────────────────────────

print('\n=== 5. Letterboxd Top 250 ===')
LB_BASE = 'https://letterboxd.com/dave/list/official-top-250-narrative-feature-films'


def fetch_lb_page(page_num):
    cache_file = CACHE_DIR / f'letterboxd_p{page_num}.html'
    if cache_file.exists():
        content = cache_file.read_text(encoding='utf-8', errors='replace')
        return content
    url = f'{LB_BASE}/page/{page_num}/' if page_num > 1 else f'{LB_BASE}/'
    print(f'  Fetching LB page {page_num}...')
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
        if resp.status_code == 403:
            cache_file.write_text('403', encoding='utf-8')
            return '403'
        resp.raise_for_status()
        html = resp.text
        cache_file.write_text(html, encoding='utf-8')
        time.sleep(2.0)
        return html
    except Exception as e:
        print(f'  Error: {e}')
        return '403'


def parse_lb_page(html):
    if html == '403':
        return []
    soup = BeautifulSoup(html, 'lxml')
    films = []
    # Primary: li.poster-container with img alt
    for item in soup.select('li.poster-container'):
        img = item.select_one('img')
        title = img.get('alt', '').strip() if img else ''
        if not title:
            for sel in ['span.frame-title', '[class*="name"]', 'a[href]']:
                el = item.select_one(sel)
                if el:
                    title = el.get_text(strip=True) or el.get('data-film-name', '')
                    if title:
                        break
        if not title:
            title = item.get('data-film-name', '')
        if title:
            year_attr = item.get('data-film-release-year', '') or ''
            year = int(year_attr) if str(year_attr).isdigit() else None
            films.append({'title': title, 'year': year})
    # Fallback: .film-poster divs
    if not films:
        for item in soup.select('[data-film-name], .film-poster[data-film-name]'):
            title = item.get('data-film-name', '').strip()
            if title:
                films.append({'title': title, 'year': None})
    return films


def fetch_imdb_lb_fallback():
    films = []
    for start in [1, 51, 101, 151, 201]:
        url = (f'https://www.imdb.com/list/ls062992231/'
               f'?start={start}&view=detail&sort=listorian:asc')
        cache = CACHE_DIR / f'imdb_lb_{start}.html'
        if cache.exists():
            html = cache.read_text(encoding='utf-8', errors='replace')
        else:
            print(f'  Fetching IMDB fallback start={start}...')
            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
                if resp.status_code != 200:
                    print(f'  Got {resp.status_code}, skipping')
                    continue
                html = resp.text
                cache.write_text(html, encoding='utf-8')
                time.sleep(2.0)
            except Exception as e:
                print(f'  Error: {e}')
                continue
        soup = BeautifulSoup(html, 'lxml')
        for item in soup.select('div.lister-item'):
            h3 = item.select_one('h3.lister-item-header a')
            yr_span = item.select_one('span.lister-item-year')
            if h3:
                title = h3.get_text(strip=True)
                yr_m = re.search(r'\d{4}', yr_span.get_text() if yr_span else '')
                year = int(yr_m.group()) if yr_m else None
                films.append({'title': title, 'year': year})
    return films


lb_films = []
use_fallback = False
for pg in range(1, 4):
    html = fetch_lb_page(pg)
    if html == '403':
        use_fallback = True
        break
    parsed = parse_lb_page(html)
    lb_films.extend(parsed)
    print(f'  Page {pg}: {len(parsed)} films')

if use_fallback or len(lb_films) < 50:
    print('Using IMDB fallback...')
    lb_films = fetch_imdb_lb_fallback()

lb_map = {}
if lb_films:
    lb_df = pd.DataFrame(lb_films)
    lb_df['lb_rank']    = range(1, len(lb_df) + 1)
    lb_df['title_norm'] = lb_df['title'].apply(normalize_title)
    print(f'Letterboxd: {len(lb_df):,} films')
    print(lb_df[['lb_rank', 'title', 'year']].head(10).to_string(index=False))
    print('\nMatching Letterboxd to corpus...')
    lb_map = match_list_to_corpus(lb_df, corpus,
                                   title_col='title_norm',
                                   year_col='year',
                                   rank_col='lb_rank')
    print(f'Letterboxd in corpus: {len(lb_map):,} works')
else:
    print('No Letterboxd/IMDB data available')


# ── 6. Assemble prestige signals ──────────────────────────────────────────────

print('\n=== 6. Assemble prestige signals ===')
prestige = corpus[['work_id', 'qid', 'title', 'medium', 'year']].copy()
prestige = prestige.merge(mc_scores, on='qid', how='left')
prestige = prestige.merge(rt_scores, on='qid', how='left')
if not wins_agg.empty:
    prestige = prestige.merge(wins_agg, on='qid', how='left')
else:
    prestige['wins_oscar'] = np.nan
    prestige['wins_bafta'] = np.nan
if not noms_agg.empty:
    prestige = prestige.merge(noms_agg, on='qid', how='left')
else:
    prestige['noms_oscar'] = np.nan
    prestige['noms_bafta'] = np.nan

# 0 awards = NaN (no signal, not bottom percentile)
for col in ['wins_oscar', 'wins_bafta', 'noms_oscar', 'noms_bafta']:
    if col in prestige.columns:
        prestige[col] = prestige[col].replace(0, np.nan)

prestige['tspdt_rank']      = prestige['work_id'].map(tspdt_map)
prestige['letterboxd_rank'] = prestige['work_id'].map(lb_map)

signal_cols = [c for c in ['metacritic', 'rt_score', 'wins_oscar', 'wins_bafta',
                             'noms_oscar', 'noms_bafta', 'tspdt_rank', 'letterboxd_rank']
               if c in prestige.columns]

print('Signal coverage:')
for col in signal_cols:
    n = prestige[col].notna().sum()
    print(f'  {col:20s}: {n:4d} ({100*n/len(prestige):.1f}%)')

prestige[['work_id','qid','title','medium','year'] + signal_cols].to_csv(
    DATA_DIR / 'prestige_signals.csv', index=False
)
print(f'Saved prestige_signals.csv')

# Key film spot check
spot_titles = 'Lawrence|Schindler|Martian|Elvis|Zero Dark|Casablanca|Gone with|Citizen Kane'
spot = prestige[prestige['title'].str.contains(spot_titles, case=False, na=False)]
print('\nKey film prestige signals:')
print(spot[['title','year','metacritic','rt_score','wins_oscar','noms_oscar',
            'tspdt_rank','letterboxd_rank']].to_string(index=False,
            float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '?'))


# ── 7. Rebuild importance_score ───────────────────────────────────────────────

print('\n=== 7. Rebuild importance_score ===')
corp = corpus.copy()
corp = corp.drop(columns=['importance_score', 'n_importance_signals', 'list_count'],
                  errors='ignore')
# Drop pre-existing signal columns before re-merging to avoid _x/_y conflicts on re-runs
corp = corp.drop(columns=signal_cols, errors='ignore')
corp = corp.merge(prestige[['work_id'] + signal_cols], on='work_id', how='left')

# Clean year column — strip Wikipedia citation markers like '1975[b]'
corp['year'] = pd.to_numeric(
    corp['year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
)

# Rank columns (invert: lower rank = higher pctile)
RANK_COLS = {
    'afi_rank'       : 'pctile_afi',
    'ss_rank'        : 'pctile_ss',
    'imdb_rank'      : 'pctile_imdb',
    'ml_rank'        : 'pctile_ml',
    'tspdt_rank'     : 'pctile_tspdt',
    'letterboxd_rank': 'pctile_lb',
}
for rank_col, pctile_col in RANK_COLS.items():
    if rank_col in corp.columns and corp[rank_col].notna().any():
        corp[pctile_col] = pctile(corp[rank_col].max() - corp[rank_col])

# Score columns (higher = better)
SCORE_COLS = {
    'ebert_great'        : 'pctile_ebert',
    'criterion'          : 'pctile_criterion',
    'time_100'           : 'pctile_time',
    'pulitzer'           : 'pctile_pulitzer',
    'booker'             : 'pctile_booker',
    'national_book_award': 'pctile_nba',
    'metacritic'         : 'pctile_mc',
    'rt_score'           : 'pctile_rt',
    'wins_oscar'         : 'pctile_oscar_wins',
    'wins_bafta'         : 'pctile_bafta_wins',
    'noms_oscar'         : 'pctile_oscar_noms',
    'noms_bafta'         : 'pctile_bafta_noms',
    'sitelinks'          : 'pctile_sitelinks',
}
for src_col, pctile_col in SCORE_COLS.items():
    if src_col in corp.columns and corp[src_col].notna().any():
        corp[pctile_col] = pctile(corp[src_col])

pctile_cols_all = [c for c in corp.columns if c.startswith('pctile_')]

# Signal-group classification notes:
#   pctile_ss   DROPPED — ss_rank is a popularity list (Titanic=#1), NOT Sight & Sound critics' poll
#   pctile_imdb MOVED to CRITIC_P — IMDb is a user rating scale, not a curated prestige list
PRESTIGE_LIST_P = [c for c in pctile_cols_all
                   if any(x in c for x in ['afi','ebert','criterion',
                                             'time','ml','pulitzer','booker','nba',
                                             'tspdt','lb'])]
AWARDS_P        = [c for c in pctile_cols_all if any(x in c for x in ['oscar','bafta'])]
CRITIC_P        = [c for c in pctile_cols_all if any(x in c for x in ['_mc','_rt','_imdb'])]
SITELINKS_P     = [c for c in pctile_cols_all if 'sitelinks' in c]

print(f'Prestige list cols ({len(PRESTIGE_LIST_P)}): {PRESTIGE_LIST_P}')
print(f'Awards cols        ({len(AWARDS_P)}): {AWARDS_P}')
print(f'Critic cols        ({len(CRITIC_P)}): {CRITIC_P}')
print(f'Sitelinks cols     ({len(SITELINKS_P)}): {SITELINKS_P}')


def group_mean(row, cols):
    vals = [row[c] for c in cols if pd.notna(row.get(c))]
    return np.mean(vals) if vals else np.nan


def compute_importance(row):
    year = row.get('year')
    is_new = pd.notna(year) and float(year) >= 2010
    if is_new:
        weights = {'prestige': 0.20, 'awards': 0.30, 'critics': 0.35, 'sitelinks': 0.15}
    else:
        weights = {'prestige': 0.35, 'awards': 0.25, 'critics': 0.25, 'sitelinks': 0.15}
    groups = {
        'prestige' : group_mean(row, PRESTIGE_LIST_P),
        'awards'   : group_mean(row, AWARDS_P),
        'critics'  : group_mean(row, CRITIC_P),
        'sitelinks': group_mean(row, SITELINKS_P),
    }
    active = {k: v for k, v in groups.items() if pd.notna(v)}
    if not active:
        return np.nan
    # No renormalization: missing groups contribute 0, not "excluded from average".
    # This naturally rewards breadth — a film with 4 groups at 70 each scores 70,
    # while one with only sitelinks at 95 scores 0.15*95=14.25 (post-2010) or
    # 0.15*95=14.25 (pre-2010).  Prevents single-signal inflation.
    return round(sum(weights[k] * v for k, v in active.items()), 4)


corp['importance_score']     = corp.apply(compute_importance, axis=1)
corp['n_importance_signals'] = corp[pctile_cols_all].notna().sum(axis=1)

print('\nBefore → After (key films):')
old_imp = corpus.set_index('work_id')['importance_score']
for t in ['Lawrence of Arabia', "Schindler's List", 'The Martian', 'Elvis',
          'Zero Dark Thirty', 'Casablanca', 'Gone with the Wind', 'Citizen Kane',
          'Raging Bull', 'My Name Is Khan']:
    row = corp[corp['title'].str.contains(t, case=False, na=False)]
    if len(row):
        r = row.iloc[0]
        old = old_imp.get(r['work_id'], float('nan'))
        no  = r.get('noms_oscar', float('nan'))
        ts  = r.get('tspdt_rank', float('nan'))
        mc  = r.get('metacritic', float('nan'))
        mc_s  = f'{mc:.0f}'  if pd.notna(mc)  else '?'
        no_s  = f'{no:.0f}'  if pd.notna(no)  else '0'
        ts_s  = f'{ts:.0f}'  if pd.notna(ts)  else '?'
        old_s = f'{old:.1f}' if pd.notna(old) else '?'
        print(f"  {r['title'][:40]:<40}  {old_s} → {r['importance_score']:.1f}"
              f"  mc={mc_s}  oscar_noms={no_s}  tspdt={ts_s}")


# ── 8. Final rankings ─────────────────────────────────────────────────────────

print('\n=== 8. Final Hirsch-calibrated rankings ===')
probs = pd.read_csv(DATA_DIR / 'hirsch_calibrated_concepts.csv')
prob_lookup = dict(zip(probs['concept'], probs['hirsch_prob']))

tags = pd.read_csv(DATA_DIR / 'work_concept_tags.csv')
matched_tags = tags[tags['matched_concept'].notna()].copy()
matched_tags['hirsch_prob']    = matched_tags['matched_concept'].map(prob_lookup).fillna(0.0)
matched_tags['hirsch_contrib'] = matched_tags['relevance'] * matched_tags['hirsch_prob']

hirsch_yield = (
    matched_tags.groupby('work_id')
    .agg(n_matched=('matched_concept', 'nunique'),
         hirsch_yield=('hirsch_contrib', 'sum'),
         avg_hirsch_prob=('hirsch_prob', 'mean'))
    .reset_index()
)

compare = hirsch_yield.merge(
    corp[['work_id', 'title', 'medium', 'year', 'importance_score']],
    on='work_id', how='inner'
)
compare['hyield_norm']     = 100 * compare['hirsch_yield']     / compare['hirsch_yield'].max()
compare['imp_norm']        = 100 * compare['importance_score'] / compare['importance_score'].max()
compare['hirsch_combined'] = 0.5 * compare['imp_norm'] + 0.5 * compare['hyield_norm']
compare = compare.sort_values('hirsch_combined', ascending=False).reset_index(drop=True)
compare['rank'] = compare.index + 1

print('\nTOP 40 FILMS:')
films = compare[compare['medium'] == 'film'].head(40)
for _, r in films.iterrows():
    yr = str(int(float(r['year']))) if pd.notna(r['year']) else '?'
    print(f"  {int(r['rank']):3d}.  {r['title'][:50]:<50}  ({yr})  "
          f"imp={r['importance_score']:.1f}  combined={r['hirsch_combined']:.1f}")

print('\nTOP 20 NOVELS:')
novels = compare[compare['medium'] == 'novel'].head(20)
for _, r in novels.iterrows():
    yr = str(int(float(r['year']))) if pd.notna(r['year']) else '?'
    print(f"  {int(r['rank']):3d}.  {r['title'][:50]:<50}  ({yr})  imp={r['importance_score']:.1f}")


# ── 9. Export ─────────────────────────────────────────────────────────────────

print('\n=== 9. Export ===')
old_pctile_cols = [c for c in corpus.columns if c.startswith('pctile_')]
new_corpus = corpus.drop(
    columns=['importance_score', 'n_importance_signals', 'list_count']
            + old_pctile_cols + signal_cols,
    errors='ignore'
)
new_corpus = new_corpus.merge(
    corp[['work_id', 'importance_score', 'n_importance_signals'] + signal_cols + pctile_cols_all],
    on='work_id', how='left'
)
new_corpus.to_csv(DATA_DIR / 'works_corpus.csv', index=False)
print(f'Saved works_corpus.csv: {len(new_corpus):,} rows, {len(new_corpus.columns)} cols')

compare.to_csv(DATA_DIR / 'works_hirsch_scored.csv', index=False)
print(f'Saved works_hirsch_scored.csv: {len(compare):,} works')

f1 = compare[compare['medium'] == 'film'].iloc[0]
n1 = compare[compare['medium'] == 'novel'].iloc[0]
print(f'\n#1 Film : {f1["title"]} (imp={f1["importance_score"]:.1f})')
print(f'#1 Novel: {n1["title"]} (imp={n1["importance_score"]:.1f})')
print('\nDone.')
