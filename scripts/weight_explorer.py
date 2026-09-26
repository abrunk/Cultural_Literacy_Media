"""
weight_explorer.py — Test different importance/scoring weights and show
where target films land under each scheme.

Run:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  .venv/Scripts/python.exe scripts/weight_explorer.py
"""
import sys, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = __import__('pathlib').Path('data')

# ── Load data ─────────────────────────────────────────────────────────────────
corpus = pd.read_csv(DATA_DIR / 'works_corpus.csv')
tags   = pd.read_csv(DATA_DIR / 'work_concept_tags.csv')
probs  = pd.read_csv(DATA_DIR / 'hirsch_calibrated_concepts.csv')

prob_lookup = dict(zip(probs['concept'], probs['hirsch_prob']))

# Hirsch concept yield (fixed — doesn't change with weights)
matched = tags[tags['matched_concept'].notna()].copy()
matched['hirsch_prob']    = matched['matched_concept'].map(prob_lookup).fillna(0.0)
matched['hirsch_contrib'] = matched['relevance'] * matched['hirsch_prob']
hirsch_yield = (
    matched.groupby('work_id')
    .agg(n_matched=('matched_concept','nunique'),
         hirsch_yield=('hirsch_contrib','sum'))
    .reset_index()
)

films = corpus[corpus['medium'] == 'film'].copy()
films['year_num'] = pd.to_numeric(
    films['year'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
)
films = films.merge(hirsch_yield, on='work_id', how='left')
films['hirsch_yield'] = films['hirsch_yield'].fillna(0.0)

# Identify pctile columns present in corpus
pctile_cols = [c for c in corpus.columns if c.startswith('pctile_')]
PRESTIGE_LIST_P = [c for c in pctile_cols if any(x in c for x in
    ['afi','ebert','criterion','time','ml','pulitzer','booker','nba','tspdt','lb'])]
# pctile_ss dropped (popularity list, not S&S critics); pctile_imdb moved to CRITIC_P
AWARDS_P        = [c for c in pctile_cols if any(x in c for x in ['oscar','bafta'])]
CRITIC_P        = [c for c in pctile_cols if any(x in c for x in ['_mc','_rt','_imdb'])]
SITELINKS_P     = [c for c in pctile_cols if 'sitelinks' in c]

def group_mean(row, cols):
    vals = [row[c] for c in cols if pd.notna(row.get(c))]
    return np.mean(vals) if vals else np.nan

def compute_imp(row, w_pre, w_new):
    year = row.get('year_num')
    is_new = pd.notna(year) and float(year) >= 2010
    w = w_new if is_new else w_pre
    groups = {
        'prestige' : group_mean(row, PRESTIGE_LIST_P),
        'awards'   : group_mean(row, AWARDS_P),
        'critics'  : group_mean(row, CRITIC_P),
        'sitelinks': group_mean(row, SITELINKS_P),
    }
    active = {k: v for k, v in groups.items() if pd.notna(v)}
    if not active:
        return np.nan
    # No renormalization: missing groups contribute 0, rewarding breadth over single-signal inflation.
    return sum(w[k] * v for k, v in active.items())

def score_scheme(label, w_pre, w_new, imp_blend):
    """Compute combined score; imp_blend = fraction going to importance (vs hirsch_yield)."""
    df = films.copy()
    df['imp'] = df.apply(lambda r: compute_imp(r, w_pre, w_new), axis=1)

    # Normalise each component 0-100
    imp_max = df['imp'].max()
    hy_max  = df['hirsch_yield'].max()
    df['imp_norm'] = 100 * df['imp'] / imp_max
    df['hy_norm']  = 100 * df['hirsch_yield'] / hy_max
    df['combined'] = imp_blend * df['imp_norm'] + (1 - imp_blend) * df['hy_norm']
    df['film_rank'] = df['combined'].rank(ascending=False, method='min').astype(int)

    return df[['work_id','title','year_num','imp','imp_norm','hy_norm','combined','film_rank']].copy()


# ── Define schemes ────────────────────────────────────────────────────────────
# w_pre / w_new dicts: prestige, awards, critics, sitelinks
SCHEMES = [
    dict(
        label    = 'A  Current baseline',
        w_pre    = dict(prestige=0.35, awards=0.25, critics=0.25, sitelinks=0.15),
        w_new    = dict(prestige=0.20, awards=0.30, critics=0.35, sitelinks=0.15),
        imp_blend= 0.50,
    ),
    dict(
        label    = 'B  Prestige lists heavier (pre-2010)',
        w_pre    = dict(prestige=0.50, awards=0.25, critics=0.15, sitelinks=0.10),
        w_new    = dict(prestige=0.25, awards=0.30, critics=0.35, sitelinks=0.10),
        imp_blend= 0.50,
    ),
    dict(
        label    = 'C  Importance dominant blend (70/30)',
        w_pre    = dict(prestige=0.35, awards=0.25, critics=0.25, sitelinks=0.15),
        w_new    = dict(prestige=0.20, awards=0.30, critics=0.35, sitelinks=0.15),
        imp_blend= 0.70,
    ),
    dict(
        label    = 'D  Prestige heavy + importance dominant',
        w_pre    = dict(prestige=0.50, awards=0.25, critics=0.15, sitelinks=0.10),
        w_new    = dict(prestige=0.25, awards=0.35, critics=0.30, sitelinks=0.10),
        imp_blend= 0.65,
    ),
    dict(
        label    = 'E  Awards + prestige, low critics',
        w_pre    = dict(prestige=0.45, awards=0.35, critics=0.10, sitelinks=0.10),
        w_new    = dict(prestige=0.25, awards=0.45, critics=0.20, sitelinks=0.10),
        imp_blend= 0.60,
    ),
]

# ── Target + noise films ──────────────────────────────────────────────────────
TARGET_PATTERNS = {
    'Lawrence of Arabia'  : 'Lawrence',
    "Schindler's List"    : 'Schindler',
    'Gone with the Wind'  : 'Gone with',
    'Zero Dark Thirty'    : 'Zero Dark',
    '12 Years a Slave'    : '12 Years',
    'Ben-Hur'             : 'Ben-Hur',
    'Spartacus'           : 'Spartacus',
}
NOISE_PATTERNS = {
    'Harry Potter (any)'  : 'Harry Potter and the Philosopher',
    'Almost Famous'       : 'Almost Famous',
    'The Dark Knight'     : 'Dark Knight',
    'The Avengers'        : 'Avengers$|^The Avengers',
    'Avengers: Age'       : 'Age of Ultron',
    'Interstellar'        : 'Interstellar',
    'Designing Woman'     : 'Designing Woman',
}

def find_rank(df, pattern):
    row = df[df['title'].str.contains(pattern, case=False, na=False, regex=True)]
    if len(row) == 0:
        return None
    return int(row.iloc[0]['film_rank'])

# ── Run and display ───────────────────────────────────────────────────────────
print('=' * 90)
print(f'{"Film":<28}', end='')
for s in SCHEMES:
    print(f'  {s["label"][:16]:>16}', end='')
print()
print('-' * 90)

results = []
for s in SCHEMES:
    results.append(score_scheme(**s))

all_patterns = {**TARGET_PATTERNS, **NOISE_PATTERNS}

for name, pat in TARGET_PATTERNS.items():
    print(f'  ✅ {name:<24}', end='')
    for r in results:
        rank = find_rank(r, pat)
        cell = f'#{rank}' if rank else '?'
        print(f'  {cell:>16}', end='')
    print()

print()
for name, pat in NOISE_PATTERNS.items():
    print(f'  ⚠️  {name:<24}', end='')
    for r in results:
        rank = find_rank(r, pat)
        cell = f'#{rank}' if rank else '?'
        print(f'  {cell:>16}', end='')
    print()

# ── Alignment score: % of target films in top 20 ─────────────────────────────
print()
print('Alignment score (target films in top 20):')
for i, s in enumerate(SCHEMES):
    in_top20 = sum(
        1 for pat in TARGET_PATTERNS.values()
        if (find_rank(results[i], pat) or 999) <= 20
    )
    print(f'  {s["label"]:<42}  {in_top20}/{len(TARGET_PATTERNS)} target films in top 20')

# ── Show top 20 for best-alignment scheme ─────────────────────────────────────
best_idx = max(
    range(len(SCHEMES)),
    key=lambda i: sum(
        1 for pat in TARGET_PATTERNS.values()
        if (find_rank(results[i], pat) or 999) <= 20
    )
)
best_scheme = SCHEMES[best_idx]
best_df = results[best_idx].sort_values('combined', ascending=False).head(20)
print(f'\nTop 20 under best-alignment scheme ({best_scheme["label"]}):')
for rank_i, (_, r) in enumerate(best_df.iterrows(), 1):
    yr = str(int(r['year_num'])) if pd.notna(r['year_num']) else '?'
    flag = '✅' if any(
        r['title'].lower().find(p.lower().split('|')[0].lstrip('^')) >= 0
        for p in TARGET_PATTERNS.values()
    ) else '  '
    print(f'  {rank_i:2d}. {flag} {r["title"][:50]:<50}  ({yr})  combined={r["combined"]:.1f}')

# ── Spot-check key signal values for context ─────────────────────────────────
print('\nKey signal spot-check (raw pctile values):')
spot_pats = {**TARGET_PATTERNS, **{k: v for k, v in NOISE_PATTERNS.items()
                                   if k not in ['Avengers: Age','Designing Woman']}}
display_cols = ['title'] + PRESTIGE_LIST_P[:4] + AWARDS_P[:2] + CRITIC_P + SITELINKS_P
for name, pat in spot_pats.items():
    row = films[films['title'].str.contains(pat, case=False, na=False, regex=True)]
    if len(row):
        r = row.iloc[0]
        vals = []
        for c in PRESTIGE_LIST_P + AWARDS_P + CRITIC_P + SITELINKS_P:
            if pd.notna(r.get(c)):
                vals.append(f'{c.replace("pctile_","")}={r[c]:.0f}')
        print(f'  {name:<24}: {", ".join(vals) if vals else "no signals"}')
