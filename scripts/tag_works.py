"""
tag_works.py — standalone pre-caching script for Notebook 10.

Run with:
  cd C:/Users/Alexander/Projects/Cultural_Literacy_Media
  PYTHONPATH=.venv/Lib/site-packages python scripts/tag_works.py

Writes one JSON file per work to data/cache/nb10/<work_id>.json
All subsequent notebook runs read from cache and complete in seconds.
"""

import pandas as pd
import numpy as np
import re
import json
import time
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

DATA_DIR  = Path('data')
CACHE_DIR = DATA_DIR / 'cache' / 'nb10'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL        = 'claude-haiku-4-5-20251001'
MAX_TOKENS   = 1024
MAX_CONCEPTS = 20
MAX_WORKERS  = 8


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


SYSTEM_PROMPT = """\
You are a cultural literacy expert. Given a film or novel, identify the cultural literacy
concepts it teaches — things a well-educated person should know to understand the work
and its place in culture.

Focus on:
- Historical and mythological figures (politicians, generals, scientists, writers, gods)
- Historical events, wars, revolutions, and periods
- Canonical literary/mythological works referenced or alluded to
- Culturally significant places (cities, regions, nations)
- Philosophical, religious, or ideological movements
- Scientific or economic concepts with broad cultural resonance
- Famous phrases, allusions, or archetypes the work embodies

Return a JSON object with a single key "concepts" whose value is an array of objects.
Each object has exactly:
  "concept"   : canonical name (e.g. "World War II", "Greek tragedy", "Sigmund Freud")
  "type"      : one of: person | place | event | theme | work | phrase | movement | other
  "relevance" : integer 1 (background reference), 2 (significant), or 3 (central)

Return at most 20 concepts. Return ONLY the JSON object, no other text."""


def build_prompt(row):
    medium = row['medium']
    title  = row['title']
    try:
        year = int(float(str(row['year']))) if pd.notna(row.get('year')) else None
    except (ValueError, TypeError):
        year = None

    lines = []
    if medium == 'film':
        lines.append(f'Film: {title}')
        if year:
            lines.append(f'Year: {year}')
        director = str(row.get('director', '')).strip()
        if director and director != 'nan':
            lines.append(f'Director: {director}')
        genre = str(row.get('genre', '')).strip()
        if genre and genre != 'nan':
            lines.append(f'Genre: {genre}')
    else:
        lines.append(f'Novel: {title}')
        if year:
            lines.append(f'Year: {year}')
        author = str(row.get('author', '')).strip()
        if author and author != 'nan':
            lines.append(f'Author: {author}')

    desc = str(row.get('wd_desc', '')).strip()
    if desc and desc != 'nan':
        lines.append(f'Description: {desc}')

    return '\n'.join(lines)


client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def tag_work(row, max_retries=3):
    work_id    = row['work_id']
    cache_path = CACHE_DIR / f'{work_id}.json'

    if cache_path.exists():
        return work_id, 'cached'

    try:
        prompt = build_prompt(row)
    except Exception as e:
        return work_id, f'prompt_error: {e}'

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            )
            raw  = msg.content[0].text.strip()
            raw  = re.sub(r'^```(?:json)?\s*', '', raw)
            raw  = re.sub(r'\s*```$', '', raw)
            data = json.loads(raw)
            if 'concepts' not in data:
                data = {'concepts': data if isinstance(data, list) else []}
            data['concepts'] = data['concepts'][:MAX_CONCEPTS]
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            return work_id, 'ok'

        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    if 'concepts' not in data:
                        data = {'concepts': []}
                    with open(cache_path, 'w') as f:
                        json.dump(data, f)
                    return work_id, 'ok_extracted'
                except Exception:
                    pass
            if attempt == max_retries - 1:
                err = {'concepts': [], 'error': 'json_decode_failed'}
                with open(cache_path, 'w') as f:
                    json.dump(err, f)
                return work_id, 'json_error'

        except anthropic.RateLimitError:
            print(f'  [{work_id}] Rate limit — sleeping 60s')
            time.sleep(60)

        except anthropic.APIStatusError as e:
            if e.status_code == 400:
                err = {'concepts': [], 'error': f'api_400: {str(e)[:80]}'}
                with open(cache_path, 'w') as f:
                    json.dump(err, f)
                return work_id, 'api_400'
            time.sleep(2 ** attempt)

        except Exception as e:
            if attempt == max_retries - 1:
                err = {'concepts': [], 'error': str(e)[:120]}
                with open(cache_path, 'w') as f:
                    json.dump(err, f)
                return work_id, f'error: {e}'
            time.sleep(2 ** attempt)

    return work_id, 'failed'


def main():
    works = pd.read_csv(DATA_DIR / 'works_corpus.csv')
    print(f'Works corpus: {len(works):,} works')

    already = sum(1 for wid in works['work_id'] if (CACHE_DIR / f'{wid}.json').exists())
    print(f'Already cached: {already:,} / {len(works):,}')
    print(f'To tag        : {len(works) - already:,}')
    print(f'Workers       : {MAX_WORKERS}')

    if already == len(works):
        print('All works already cached. Done.')
        return

    work_rows = [row for _, row in works.iterrows()]
    t0   = time.time()
    done = 0
    ok   = 0
    errs = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(tag_work, row): row['work_id'] for row in work_rows}
        for future in as_completed(futures):
            work_id, status = future.result()
            done += 1
            if status in ('ok', 'ok_extracted', 'cached'):
                ok += 1
            else:
                errs.append((work_id, status))

            if done % 50 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(works) - done) / rate
                print(f'  [{done:4d}/{len(works)}] {elapsed:5.0f}s  '
                      f'rate={rate:.1f}/s  ETA~{eta:.0f}s  ok={ok}  err={len(errs)}')

    elapsed = time.time() - t0
    print(f'\nDone. {done:,} works in {elapsed:.0f}s  ({elapsed/done:.2f}s/work)')
    cached_now = sum(1 for wid in works['work_id'] if (CACHE_DIR / f'{wid}.json').exists())
    print(f'Cache files: {cached_now:,} / {len(works):,}')
    if errs:
        print(f'\nErrors ({len(errs)}):')
        for wid, st in errs[:20]:
            print(f'  {wid}: {st}')


if __name__ == '__main__':
    main()
