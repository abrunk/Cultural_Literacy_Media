"""
domain_sublists.py — Within-domain film rankings for content area sub-lists.

Uses manually curated film lists per domain so there are no false positives.
Ranks within each domain using critics/awards signals (not prestige lists, which
are biased toward classic films and penalise post-2000 releases).

Run:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/domain_sublists.py
"""
import sys, warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache' / 'nb15'

# ── Load data ──────────────────────────────────────────────────────────────────
corpus = pd.read_csv(DATA_DIR / 'works_corpus.csv')
scored = pd.read_csv(DATA_DIR / 'works_hirsch_scored.csv')
films  = corpus[corpus['medium'].isin(['film', 'miniseries'])].copy()
p921_raw = json.loads((CACHE_DIR / 'p921_raw.json').read_text(encoding='utf-8'))

# Fix NaN QIDs before building lookup
films['qid_str'] = films['qid'].apply(
    lambda q: str(q).split('/')[-1] if pd.notna(q) else ''
)

# P921 subject lookup: qid_str -> list of lowercase subject labels
from collections import defaultdict
qid_subs = defaultdict(list)
for r in p921_raw:
    qid_subs[r['qid']].append(r.get('subjectLabel', '').lower())

# Quality pctile columns — contemporary signals only (no classic-list bias)
QUALITY_COLS   = ['pctile_mc', 'pctile_rt', 'pctile_imdb']
AWARDS_COLS    = ['pctile_oscar_wins', 'pctile_oscar_noms', 'pctile_bafta_wins']
CULTURAL_COLS  = ['pctile_sitelinks']


def quality_score(row):
    """Mean of available critic + awards + cultural pctiles. Treats missing as absent."""
    vals = []
    for c in QUALITY_COLS:
        v = row.get(c)
        if pd.notna(v):
            vals.append(float(v))
    # Awards count half-weight
    for c in AWARDS_COLS:
        v = row.get(c)
        if pd.notna(v):
            vals.append(float(v) * 0.5)
    for c in CULTURAL_COLS:
        v = row.get(c)
        if pd.notna(v):
            vals.append(float(v) * 0.3)
    return round(np.mean(vals), 1) if vals else None


def domain_depth(qid_str, terms):
    """Fraction of a film's P921 subjects that match domain terms."""
    subs = qid_subs.get(qid_str, [])
    if not subs:
        return None
    hits = sum(1 for s in subs if any(t in s for t in terms))
    return round(hits / len(subs), 2)


def lookup(title_fragment):
    """Find a film row by partial title match (case-insensitive, literal).
    Prefers exact title matches over partial matches to avoid false positives."""
    mask = films['title'].str.contains(title_fragment, case=False, na=False, regex=False)
    hits = films[mask]
    if len(hits) == 0:
        return None
    exact = hits[hits['title'].str.lower() == title_fragment.lower()]
    return exact.iloc[0] if len(exact) else hits.iloc[0]


def get_scores(title_fragment):
    """Return a dict of useful scores for a film, or None if not in corpus."""
    r = lookup(title_fragment)
    if r is None:
        return None
    sc = scored[scored['work_id'] == r['work_id']]
    rank    = int(sc.iloc[0]['rank'])          if len(sc) else None
    enr     = sc.iloc[0]['enr_combined']       if len(sc) else None
    imp     = r.get('importance_score')
    qs      = quality_score(r)
    qid_str = r['qid_str']
    dd      = domain_depth(qid_str, [])        # domain-specific, set per domain
    p_mc    = r.get('pctile_mc')
    p_rt    = r.get('pctile_rt')
    p_imdb  = r.get('pctile_imdb')
    p_osc   = r.get('pctile_oscar_wins')
    p_baf   = r.get('pctile_bafta_wins')
    p_sl    = r.get('pctile_sitelinks')
    return dict(
        title   = r['title'],
        year    = str(r['year'])[:4] if pd.notna(r.get('year')) else '?',
        rank    = rank,
        enr     = enr,
        imp     = round(float(imp), 1) if pd.notna(imp) else None,
        qs      = qs,
        mc      = int(p_mc)   if pd.notna(p_mc)   else None,
        rt      = int(p_rt)   if pd.notna(p_rt)   else None,
        imdb    = int(p_imdb) if pd.notna(p_imdb) else None,
        osc     = int(p_osc)  if pd.notna(p_osc)  else None,
        bafta   = int(p_baf)  if pd.notna(p_baf)  else None,
        sl      = int(p_sl)   if pd.notna(p_sl)   else None,
        qid     = qid_str,
        n_p921  = len(qid_subs.get(qid_str, [])),
    )


def print_domain(domain_name, film_list, depth_terms):
    """
    film_list: list of (title_fragment, canonical_title, notes)
      notes: e.g. 'Iraq War', '9/11', 'documentary'
    depth_terms: list of lowercase substrings to match P921 subjects
    """
    print(f'\n{"="*90}')
    print(f'  DOMAIN: {domain_name}')
    print(f'{"="*90}')

    rows = []
    missing = []

    for frag, canon, note in film_list:
        s = get_scores(frag)
        if s is None:
            missing.append((canon, note))
            continue
        # Recompute domain depth with the right terms
        s['depth'] = domain_depth(s['qid'], depth_terms)
        s['note']  = note
        s['canon'] = canon
        rows.append(s)

    # Sort: quality_score desc, then sitelinks desc as tiebreak
    def sort_key(x):
        qs = x['qs'] if x['qs'] is not None else 0
        sl = x['sl'] if x['sl'] is not None else 0
        return (qs + 0.1 * sl)

    rows.sort(key=sort_key, reverse=True)

    # Print header
    print(f'\n  {"#":>3}  {"Title":<42} {"Yr":>4}  {"MC":>4} {"RT":>4} {"IMDb":>4}'
          f' {"Osc":>4} {"SL":>4}  {"QS":>5}  {"P921":>4}  {"Overall#":>8}  Note')
    print('  ' + '-'*105)

    for i, s in enumerate(rows, 1):
        mc   = f'{s["mc"]:3d}'   if s['mc']   is not None else '  —'
        rt   = f'{s["rt"]:3d}'   if s['rt']   is not None else '  —'
        imdb = f'{s["imdb"]:3d}' if s['imdb'] is not None else '  —'
        osc  = f'{s["osc"]:3d}'  if s['osc']  is not None else '  —'
        sl   = f'{s["sl"]:3d}'   if s['sl']   is not None else '  —'
        qs   = f'{s["qs"]:5.1f}' if s['qs']   is not None else '    —'
        rk   = f'#{s["rank"]}'   if s['rank'] is not None else '?'
        p921 = f'{s["n_p921"]:3d}' if s['n_p921'] else '  —'
        print(f'  {i:3d}  {s["title"][:41]:<42} {s["year"]:>4}  {mc:>4} {rt:>4} {imdb:>4}'
              f' {osc:>4} {sl:>4}  {qs:>5}  {p921:>4}  {rk:>8}  {s["note"]}')

    if missing:
        print(f'\n  NOT IN CORPUS ({len(missing)} films):')
        for canon, note in missing:
            print(f'    ✗  {canon:<45}  [{note}]')

    print(f'\n  {len(rows)} in corpus, {len(missing)} missing')


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 1: 9/11 & The War on Terror
# ══════════════════════════════════════════════════════════════════════════════

TERROR_FILMS = [
    # (search fragment,            canonical title,                    note)
    ('Zero Dark Thirty',           'Zero Dark Thirty',                 'bin Laden hunt'),
    ('United 93',                  'United 93',                        '9/11 – UA flight 93'),
    ('Hurt Locker',                'The Hurt Locker',                  'Iraq War – EOD'),
    ('World Trade Center',         'World Trade Center',               '9/11 – NYPD rescue'),
    ('Fahrenheit 9/11',            'Fahrenheit 9/11',                  'documentary'),
    ('Extremely Loud',             'Extremely Loud & Incredibly Close','9/11 – grief drama'),
    ('Reign Over Me',              'Reign Over Me',                    '9/11 – grief drama'),
    ("11'09",                      "11'09\"01 September 11",           'anthology short'),
    ('American Sniper',            'American Sniper',                  'Iraq War – biography'),
    ('Lone Survivor',              'Lone Survivor',                    'Afghanistan – Op Red Wings'),
    ('Charlie Wilson',             "Charlie Wilson's War",             'Afghanistan – Soviet war origin'),
    ('Green Zone',                 'Green Zone',                       'Iraq War – WMD search'),
    ('Jarhead',                    'Jarhead',                          'Gulf War / Iraq'),
    ('Body of Lies',               'Body of Lies',                     'CIA / Iraq'),
    ('Lions for Lambs',            'Lions for Lambs',                  'Afghanistan / politics'),
    ('Kite Runner',                'The Kite Runner',                  'Afghanistan – human story'),
    ('Restrepo',                   'Restrepo',                         'Afghanistan – documentary'),
    ('No End in Sight',            'No End in Sight',                  'Iraq War – documentary'),
    ('Generation Kill',            'Generation Kill',                  'Iraq War – Marines [miniseries]'),
    ('Looming Tower',              'The Looming Tower',                '9/11 / FBI-CIA rivalry [miniseries]'),
]

TERROR_DEPTH_TERMS = [
    'september 11', '9/11', 'war on terror', 'war in afghanistan',
    'invasion of afghanistan', 'war in iraq', 'iraq war', 'invasion of iraq',
    'al-qaeda', 'taliban', 'osama bin laden', 'guantanamo',
    'counterterrorism', 'counterinsurgency', 'global war on terrorism',
]

print_domain('9/11 & The War on Terror', TERROR_FILMS, TERROR_DEPTH_TERMS)


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN 2: Ancient & Classical World
# ══════════════════════════════════════════════════════════════════════════════

CLASSICAL_FILMS = [
    ('Ben-Hur',               'Ben-Hur',                       'Rome / Biblical era'),
    ('Spartacus',             'Spartacus',                     'Roman Republic / slave revolt'),
    ('Gladiator',             'Gladiator',                     'Roman Empire'),
    ('300',                   '300',                           'Battle of Thermopylae'),
    ('Troy',                  'Troy',                          'Trojan War'),
    ('Cleopatra',             'Cleopatra',                     'Ancient Egypt / Rome'),
    ('Ten Commandments',      'The Ten Commandments',          'Ancient Egypt / Moses'),
    ('Quo Vadis',             'Quo Vadis',                     'Nero / early Christianity'),
    ('Julius Caesar',         'Julius Caesar',                 'Roman Republic'),
    ('The Robe',              'The Robe',                      'Crucifixion / early Christianity'),
    ('Alexander',             'Alexander',                     'Alexander the Great'),
    ('Passion of the Christ', 'The Passion of the Christ',     'Crucifixion'),
    ('Barabbas',              'Barabbas',                      'Biblical / Roman era'),
    ('King of Kings',         'King of Kings',                 'Life of Jesus'),
    ('Greatest Story Ever',   'The Greatest Story Ever Told',  'Life of Jesus'),
    ('Fall of the Roman',     'The Fall of the Roman Empire',  'Roman Empire decline'),
    ('Agora',                 'Agora',                         'Late Roman Empire / Alexandria'),
    ('I, Claudius',           'I, Claudius',                   'Julio-Claudian dynasty [miniseries]'),
]

CLASSICAL_DEPTH_TERMS = [
    'roman empire', 'roman republic', 'ancient rome', 'ancient greece',
    'ancient egypt', 'julius caesar', 'cleopatra', 'spartacus',
    'trojan war', 'alexander the great', 'gladiat', 'third servile war',
    'crucifixion', 'early christianity', 'biblical', 'thermopylae',
    'augustus', 'nero', 'caligula', 'moses', 'exodus',
    'roman', 'greece', 'athens', 'sparta', 'carthage',
]

print_domain('Ancient & Classical World', CLASSICAL_FILMS, CLASSICAL_DEPTH_TERMS)
