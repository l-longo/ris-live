# =============================================================================
# RIS Llama-70B - live monthly update for the GitHub Pages dashboard
#
# Extends the Llama-3.3-70B Reddit Inflation Signal of
#   Del Monaco, Longo, Marcucci, Tafani - "Reddit's Pulse on US Inflation:
#   Forecasting with Large Language Models"
# beyond the end of the JAE sample (2025M8), and compares it with CPI and core
# PCE inflation, as in the descriptive chart of the paper.
#
#   1. new Reddit posts (from 2025-09-01) are screened as in the paper and
#      labelled with llama-3.3-70b-instruct on GPT@JRC (prompts of the Online
#      Appendix, temperature 0); labels are cached in labels/
#   2. daily signals are aggregated exactly as in
#      reddit_pulse_replication/_build/build_daily_signals.py (build_llama70)
#      and appended to data/daily_signals_llama70b.csv of the replication
#   3. monthly RIS variants = daily.rolling(w).mean().resample('M').mean()
#      (forecast_codes/_common.monthly_merge), w in MA_seq, 3 subreddits
#   4. CPIAUCSL / PCEPILFE 12-month log changes from FRED
#   5. docs/data/dashboard.json (+ git push with PUSH=1)
#
# Run:  python update_dashboard.py           label + build json
#       PUSH=1 python update_dashboard.py    ... and commit/push
#       MOCK_LLM=1 python update_dashboard.py   offline test, separate cache/output
# =============================================================================
import os, re, sys, json, time, subprocess
from io import StringIO
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DESK = os.path.expanduser('~/Desktop')
REDDIT_DIR = os.environ.get('RIS_REDDIT_DIR', os.path.join(HOME_DESK, 'subreddits'))
REPLICATION_DIR = os.environ.get('RIS_REPLICATION_DIR', os.path.join(HOME_DESK, 'reddit_pulse_replication'))
TOKEN_FILE = os.environ.get('GPTJRC_TOKEN_FILE', os.path.join(HOME_DESK, 'live_project', 'tokens', 'token-key.jwt'))
JRC_BASE_URL = 'https://api-gpt.jrc.ec.europa.eu/v1'
MODEL = 'llama-3.3-70b-instruct'
MOCK = os.environ.get('MOCK_LLM', '0') == '1'
PUSH = os.environ.get('PUSH', '0') == '1'
MAX_CALLS = int(os.environ.get('MAX_CALLS', '1000000'))
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
if not FRED_API_KEY:                                                          # same key as live_project, read from there (never committed)
    try:
        FRED_API_KEY = re.search(r'FRED_API_KEY\s*=\s*["\']([0-9a-f]{32})["\']', open(os.path.join(HOME_DESK, 'live_project', 'daily_forecast.py')).read()).group(1)
    except Exception:
        FRED_API_KEY = ''

LIVE_START = pd.Timestamp('2025-09-01')          # JAE sample ends 2025M8
JAE_END = pd.Timestamp('2025-08-31')
ZSCORE_START, ZSCORE_END = '2009-01-01', '2025-08-31'   # standardisation window = JAE evaluation sample, kept fixed
CHART_START = '2009-01-01'
MA_SEQ = [1, 5, 10, 30, 90, 120, 180, 360]
COMMENT_WINDOW = pd.Timedelta(days=14)
SUBREDDITS = {'Economics': 'economics', 'economy': 'economy', 'wallstreetbets': 'wallstreetbets'}   # csv name -> paper suffix
KW_ALL = r'\b(?:inflation|deflation|disinflation|hyperinflation|price|prices)\b'
KW_WSB = r'\b(?:inflation|deflation|disinflation|hyperinflation)\b'
SIGNAL_VALUE = {'UP': 1, 'DOWN': -1, 'NEUTRAL': 0}
MAX_CHARS = 6000
CHECKPOINT_EVERY = 50

LABEL_DIR = os.path.join(REPO_DIR, 'labels_mock' if MOCK else 'labels')
GEO_FILE = os.path.join(LABEL_DIR, 'geo_submissions.csv')
SIG_FILE = os.path.join(LABEL_DIR, 'signal_llama70b.csv')
FACTIVA_FILE = os.environ.get('RIS_FACTIVA_FILE', os.path.join(REDDIT_DIR, 'US_inflation.csv'))
FACTIVA_SIG_FILE = os.path.join(LABEL_DIR, 'factiva_titles_signal_llama70b.csv')
ITEMS_FILE = os.path.join(REPO_DIR, 'data', 'items_live.csv')                # texts for label_finetuned.py (never committed)
FT_DIR = os.path.join(REPO_DIR, 'labels', 'finetuned')
FT_MODELS = ['gemma2_2b_xqdora', 'llama3_2_1b_xqdora', 'llama3_2_3b_xqdora', 'qwen2_5_0_5b_xqdora', 'qwen2_5_1_5b_xqdora', 'inflabert']
FT_SUFFIX = {'Economics': 'economics', 'economy': 'economy', 'wallstreetbets': 'wsb'}
MACRO_CACHE = os.path.join(REPO_DIR, 'data', 'macro_fred_mock.csv' if MOCK else 'macro_fred.csv')
DAILY_LIVE_FILE = os.path.join(REPO_DIR, 'data', 'daily_signals_llama70b_live_mock.csv' if MOCK else 'daily_signals_llama70b_live.csv')
OUT_FILE = os.path.join(REPO_DIR, 'docs', 'data', 'dashboard_mock.json' if MOCK else 'dashboard.json')
for p in [LABEL_DIR, os.path.dirname(DAILY_LIVE_FILE), os.path.dirname(OUT_FILE)]:
    os.makedirs(p, exist_ok=True)

# Prompts: Online Appendix, fig:app:llama70b-prompt-geo and fig:app:llama70b-prompt-all-posts
GEO_PROMPT = ("You are a forecaster (linguistic interpreter) and you have to analyze the title of a social media post.\n"
              "Given the text of the title you have to classify the country that the user is referring to.\n"
              "You have to choose among one of these labels:\n"
              "US; non-US; other.\n"
              "US: the title has to be related specifically to US, not just with keywords, but also with mentions of people/institutions/events related to the US.\n"
              "non-US: the title has to be related specifically to a country other than the US.\n"
              "other: the title does not have a specific reference to a country.\n"
              "Return ONLY the classification. No punctuation at the end; no quotes.")
SIG_PROMPT = ("You are a linguistic interpreter and want to predict the future inflation rate in US from textual documents. "
              "The following document will report sentences potentially referring to US: {text}. "
              "You have to print a signal that can be UP (if the document is signalling inflation going up in the short/long-term run), "
              "DOWN (if the sentence is signalling inflation going down in the short/long-term run) or NEUTRAL "
              "(if the sentence is neutral or does not signal a particular direction on inflation). "
              "Print only the results of the signal, do not summarize the sentence nor give any reason on your choice. "
              "Even if more sentences or paragraphs are provided to you, you only have to print one signal that can be UP, DOWN or NEUTRAL.")

t0 = time.time()
print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] RIS live update (mock={MOCK})')

# ----------------------------------------------------------------------------
# 1. Reddit corpus after the JAE sample: keyword screen
# ----------------------------------------------------------------------------
subs_list, coms_list, raw_days, data_through = [], [], [], {}
for s in SUBREDDITS:
    kw = KW_WSB if s == 'wallstreetbets' else KW_ALL
    sub = pd.read_csv(os.path.join(REDDIT_DIR, f'{s}_inflation_submissions.csv'),
                      usecols=['id', 'created_utc', 'title'], dtype={'id': str})
    com = pd.read_csv(os.path.join(REDDIT_DIR, f'{s}_inflation_comments.csv'),
                      usecols=['id', 'created_utc', 'body', 'submission_id'], dtype={'id': str, 'submission_id': str})
    sub['created'] = pd.to_datetime(pd.to_numeric(sub['created_utc'], errors='coerce'), unit='s')
    com['created'] = pd.to_datetime(pd.to_numeric(com['created_utc'], errors='coerce'), unit='s')
    data_through[s] = str(max(sub['created'].max(), com['created'].max()).date())
    raw_days.append(pd.concat([sub['created'], com['created']]).dt.floor('D'))
    sub = sub[sub['created'] >= LIVE_START]                                   # threads opened after the JAE sample
    com = com[com['created'] >= LIVE_START]
    sub = sub[sub['title'].fillna('').str.contains(kw, case=False, regex=True)]
    com = com[com['body'].fillna('').str.contains(kw, case=False, regex=True)]
    com = com[~com['body'].isin(['[deleted]', '[removed]'])]
    sub['subreddit'] = s
    com['subreddit'] = s
    subs_list.append(sub)
    coms_list.append(com)
subs = pd.concat(subs_list, ignore_index=True).drop_duplicates('id')
coms = pd.concat(coms_list, ignore_index=True).drop_duplicates('id')
print(f'keyword screen: {len(subs)} submissions, {len(coms)} comments since {LIVE_START.date()}')

# ----------------------------------------------------------------------------
# 2. GPT@JRC client (same set-up as live_project/daily_forecast.py)
# ----------------------------------------------------------------------------
calls, stop_all = 0, False
if not MOCK:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=open(TOKEN_FILE).read().strip(), base_url=JRC_BASE_URL, timeout=300.0, max_retries=0)
    except Exception as e:                                                    # no client: rebuild from the cached labels only
        print(f'WARNING: GPT@JRC client not available ({type(e).__name__}: {e}) - no new labels this run')
        stop_all = True

# ----------------------------------------------------------------------------
# 3. Geographic screen of submissions (cached)
# ----------------------------------------------------------------------------
geo = pd.read_csv(GEO_FILE, dtype={'id': str}) if os.path.exists(GEO_FILE) else pd.DataFrame(columns=['id', 'subreddit', 'geo', 'raw', 'labeled_at'])
todo = subs[~subs['id'].isin(geo['id'])].sort_values('created')
print(f'geo: {len(geo)} cached, {len(todo)} to label')
new_rows = []
for row in todo.itertuples(index=False):
    if calls >= MAX_CALLS or stop_all:
        break
    title = str(row.title)
    raw = None
    if MOCK:
        raw = 'non-US' if re.search(r'\b(UK|Britain|China|Japan|Euro|Canada|Germany|India|Turkey|Argentina|Australia)\b', title) else ('US' if re.search(r'\b(US|U\.S\.|Fed|America|Powell|Trump)\b', title) else 'other')
    else:
        rl = 0
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(model=MODEL, temperature=0.0,
                                                      messages=[{'role': 'system', 'content': GEO_PROMPT}, {'role': 'user', 'content': title}])
                raw = resp.choices[0].message.content or ''
                break
            except Exception as e:
                is_429 = '429' in str(e) or 'RateLimit' in type(e).__name__
                rl = rl + 1 if is_429 else 0
                wait = min(45 * 2 ** (rl - 1), 1800) if is_429 else 10
                print(f'  geo {row.id}: {type(e).__name__} -> retry in {wait}s')
                time.sleep(wait)
        calls += 1
        if raw is None:
            print('  GPT@JRC not responding - stopping (progress saved)'); stop_all = True; break
    r_ = raw.strip().strip('."\'').lower()
    lab = 'non-US' if r_.startswith('non') else ('US' if re.match(r'^(us|u\.s\.?|usa)\b', r_) else ('other' if r_.startswith('other') else None))
    if lab is not None:
        new_rows.append({'id': row.id, 'subreddit': row.subreddit, 'geo': lab, 'raw': raw.strip()[:40],
                         'labeled_at': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    if new_rows and len(new_rows) % CHECKPOINT_EVERY == 0:
        pd.concat([geo, pd.DataFrame(new_rows)], ignore_index=True).to_csv(GEO_FILE, index=False)
        print(f'  geo {len(new_rows)}/{len(todo)}  ({time.time() - t0:.0f}s)')
geo = pd.concat([geo, pd.DataFrame(new_rows)], ignore_index=True)
geo.to_csv(GEO_FILE, index=False)

subs = subs.merge(geo[['id', 'geo']], on='id', how='inner')
subs = subs[subs['geo'].isin(['US', 'other'])]                               # keep US + no clear country
coms = coms.merge(subs[['id', 'created']].rename(columns={'id': 'submission_id', 'created': 'created_sub'}), on='submission_id', how='inner')
coms = coms[(coms['created'] >= coms['created_sub']) & (coms['created'] - coms['created_sub'] <= COMMENT_WINDOW)]
print(f'after geo screen and 14-day window: {len(subs)} submissions, {len(coms)} comments')

# ----------------------------------------------------------------------------
# 4. Directional labels UP / DOWN / NEUTRAL (cached)
# ----------------------------------------------------------------------------
items = pd.concat([
    pd.DataFrame({'id': subs['id'], 'kind': 'submission', 'id_sub': subs['id'], 'subreddit': subs['subreddit'],
                  'created': subs['created'], 'text': subs['title'].astype(str)}),
    pd.DataFrame({'id': coms['id'], 'kind': 'comment', 'id_sub': coms['submission_id'], 'subreddit': coms['subreddit'],
                  'created': coms['created'], 'text': coms['body'].astype(str)}),
], ignore_index=True)
sig = pd.read_csv(SIG_FILE, dtype={'id': str}) if os.path.exists(SIG_FILE) else pd.DataFrame(columns=['id', 'kind', 'subreddit', 'label', 'raw', 'labeled_at'])
done_keys = set(sig['id'].astype(str) + '|' + sig['kind'].astype(str))
todo = items[~(items['id'] + '|' + items['kind']).isin(done_keys)].sort_values('created')
print(f'signal: {len(sig)} cached, {len(todo)} to label')
new_rows = []
for row in todo.itertuples(index=False):
    if calls >= MAX_CALLS or stop_all:
        break
    text = row.text[:MAX_CHARS]
    raw = None
    if MOCK:
        t = text.lower()
        u = len(re.findall(r'(rising|rise|surge|soar|higher|increase|accelerat|spike|jump|tariff)', t))
        d = len(re.findall(r'(fall|falling|cool|lower|decrease|decline|easing|slow|drop|deflation)', t))
        raw = 'UP' if u > d else ('DOWN' if d > u else 'NEUTRAL')
    else:
        rl = 0
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(model=MODEL, temperature=0.0,
                                                      messages=[{'role': 'user', 'content': SIG_PROMPT.format(text=text)}])
                raw = resp.choices[0].message.content or ''
                break
            except Exception as e:
                if 'BadRequest' in type(e).__name__:                          # e.g. content filter: record, do not retry
                    raw = 'BADREQUEST'; break
                is_429 = '429' in str(e) or 'RateLimit' in type(e).__name__
                rl = rl + 1 if is_429 else 0
                wait = min(45 * 2 ** (rl - 1), 1800) if is_429 else 10
                print(f'  signal {row.id}: {type(e).__name__} -> retry in {wait}s')
                time.sleep(wait)
        calls += 1
        if raw is None:
            print('  GPT@JRC not responding - stopping (progress saved)'); stop_all = True; break
    m = re.search(r'\b(UP|DOWN|NEUTRAL)\b', raw.upper())
    new_rows.append({'id': row.id, 'kind': row.kind, 'subreddit': row.subreddit, 'label': m.group(1) if m else 'UNPARSED',
                     'raw': raw.strip()[:40], 'labeled_at': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    if len(new_rows) % CHECKPOINT_EVERY == 0:
        pd.concat([sig, pd.DataFrame(new_rows)], ignore_index=True).to_csv(SIG_FILE, index=False)
        print(f'  signal {len(new_rows)}/{len(todo)}  ({time.time() - t0:.0f}s)')
sig = pd.concat([sig, pd.DataFrame(new_rows)], ignore_index=True)
sig.to_csv(SIG_FILE, index=False)
print(f'GPT@JRC calls this run: {calls}')

# ----------------------------------------------------------------------------
# 5. Daily signals, aggregation of build_daily_signals.build_llama70:
#    thread value = mean of its items' signals (UP=1, DOWN=-1, NEUTRAL=0),
#    discretised at 0, dated at the submission date, summed by day
# ----------------------------------------------------------------------------
lab = items.merge(sig[['id', 'kind', 'label']], on=['id', 'kind'], how='left')
n_pending = int(lab['label'].isna().sum())
complete_threads = set(lab.loc[lab['kind'] == 'submission'].dropna(subset=['label'])['id_sub'])
lab['label'] = lab['label'].replace({'UNPARSED': 'NEUTRAL'})                # model asked for more text (very short posts): no direction
lab = lab[lab['label'].isin(SIGNAL_VALUE.keys())].copy()
lab = lab[lab['id_sub'].isin(complete_threads)]                              # a thread enters once its submission is labelled
lab['llama70_economics'] = lab['label'].map(SIGNAL_VALUE).astype(float)
thread = lab.groupby('id_sub').agg(date=('created', 'min'), subreddit=('subreddit', 'first'), value=('llama70_economics', 'mean'))
thread['date'] = thread['date'].dt.floor('D')
thread['disc'] = np.where(thread['value'] > 0, 1, np.where(thread['value'] < 0, -1, 0))

last_day = pd.Timestamp(min(data_through.values()))                           # last day covered by all three subreddits
live_days = pd.date_range(LIVE_START, last_day, freq='D')
covered = set(pd.concat(raw_days))
daily_live = pd.DataFrame(index=live_days)
for s, suffix in SUBREDDITS.items():
    col = f'llama70_economics_{suffix}'
    daily_live[col] = thread[thread['subreddit'] == s].groupby('date')['disc'].sum().reindex(live_days, fill_value=0).astype(float)
missing_days = [d for d in live_days if d not in covered]                     # no rows at all in the source csv: outage, not zero signal
daily_live.loc[missing_days, :] = np.nan

# ----------------------------------------------------------------------------
# 5b. Reddit outage days filled with Factiva newspaper TITLES (US_inflation.csv),
#     labelled with the same prompt; each article counts as a one-item thread.
#     Days after the last Factiva date stay missing.
# ----------------------------------------------------------------------------
factiva_days = []
if missing_days and os.path.exists(FACTIVA_FILE):
    fac = pd.read_csv(FACTIVA_FILE, usecols=['an', 'date', 'title'], dtype={'an': str}, low_memory=False)
    fac['date'] = pd.to_datetime(fac['date'], errors='coerce').dt.floor('D')
    fac = fac.dropna(subset=['date', 'title'])
    fac['title'] = fac['title'].astype(str).str.strip()
    fac = fac.drop_duplicates(['date', 'title'])
    fac_last = fac['date'].max()
    factiva_days = [d for d in missing_days if d <= fac_last]
    fac = fac[fac['date'].isin(factiva_days)].sort_values('date')
    fsig = pd.read_csv(FACTIVA_SIG_FILE, dtype={'an': str}) if os.path.exists(FACTIVA_SIG_FILE) else pd.DataFrame(columns=['an', 'date', 'label', 'raw', 'labeled_at'])
    ftodo = fac[~fac['an'].isin(fsig['an'])]
    print(f'factiva: {len(factiva_days)} outage days covered, {len(fac)} titles, {len(ftodo)} to label')
    new_rows = []
    for row in ftodo.itertuples(index=False):
        if calls >= MAX_CALLS or stop_all:
            break
        raw = None
        if MOCK:
            t = row.title.lower()
            u = len(re.findall(r'(rising|rise|surge|soar|higher|increase|accelerat|spike|jump|tariff)', t))
            d = len(re.findall(r'(fall|falling|cool|lower|decrease|decline|easing|slow|drop|deflation)', t))
            raw = 'UP' if u > d else ('DOWN' if d > u else 'NEUTRAL')
        else:
            rl = 0
            for attempt in range(8):
                try:
                    resp = client.chat.completions.create(model=MODEL, temperature=0.0,
                                                          messages=[{'role': 'user', 'content': SIG_PROMPT.format(text=row.title)}])
                    raw = resp.choices[0].message.content or ''
                    break
                except Exception as e:
                    if 'BadRequest' in type(e).__name__:
                        raw = 'BADREQUEST'; break
                    is_429 = '429' in str(e) or 'RateLimit' in type(e).__name__
                    rl = rl + 1 if is_429 else 0
                    wait = min(45 * 2 ** (rl - 1), 1800) if is_429 else 10
                    print(f'  factiva {row.an}: {type(e).__name__} -> retry in {wait}s')
                    time.sleep(wait)
            calls += 1
            if raw is None:
                print('  GPT@JRC not responding - stopping (progress saved)'); stop_all = True; break
        m = re.search(r'\b(UP|DOWN|NEUTRAL)\b', raw.upper())
        new_rows.append({'an': row.an, 'date': row.date.strftime('%Y-%m-%d'), 'label': m.group(1) if m else 'UNPARSED',
                         'raw': raw.strip()[:40], 'labeled_at': datetime.now(timezone.utc).isoformat(timespec='seconds')})
    fsig = pd.concat([fsig, pd.DataFrame(new_rows)], ignore_index=True)
    fsig.to_csv(FACTIVA_SIG_FILE, index=False)
    flab = fac.merge(fsig[['an', 'label']], on='an', how='inner')
    flab = flab[flab['label'].isin(SIGNAL_VALUE.keys())]
    fday = flab['label'].map(SIGNAL_VALUE).groupby(flab['date']).sum().reindex(factiva_days, fill_value=0).astype(float)
    if fac['an'].isin(fsig['an']).all():                                     # fill only once every title of those days is labelled
        for c in daily_live.columns:
            daily_live.loc[factiva_days, c] = fday.values
    else:
        factiva_days = []
still_missing = [d for d in missing_days if d not in set(factiva_days)]
daily_live.index.name = 'date'
daily_live.to_csv(DAILY_LIVE_FILE)

# texts for the fine-tuned classifiers (label_finetuned.py): the screened Reddit
# items plus the Factiva titles of the outage days
items_out = items[['id', 'kind', 'id_sub', 'subreddit', 'created', 'text']].copy()
if factiva_days:
    fac_out = fac[['an', 'date', 'title']].rename(columns={'an': 'id', 'date': 'created', 'title': 'text'})
    fac_out['kind'], fac_out['id_sub'], fac_out['subreddit'] = 'factiva', fac_out['id'], ''
    items_out = pd.concat([items_out, fac_out[items_out.columns]], ignore_index=True)
if not MOCK:
    items_out.to_csv(ITEMS_FILE, index=False)
if missing_days:
    print(f'Reddit outage days: {len(missing_days)}; filled with Factiva titles: {len(factiva_days)}; still missing: {len(still_missing)}')

# history of the paper + live extension
hist = pd.read_csv(os.path.join(REPLICATION_DIR, 'data', 'daily_signals_llama70b.csv'), index_col=0, parse_dates=True)
hist = hist[hist.index <= JAE_END]
daily = pd.concat([hist, daily_live[hist.columns]], axis=0)
daily = daily[~daily.index.duplicated(keep='first')].asfreq('D')

# ----------------------------------------------------------------------------
# 6. Monthly RIS variants (monthly_merge of forecast_codes/_common.py)
# ----------------------------------------------------------------------------
# In the paper file a subreddit is NaN on the last days of the sample when no
# thread was opened (concat of series ending on different days). Up to 2025M8
# the variants are computed on the file as it is, so they match df_jae_reddit;
# for the live months those trailing NaN are zero signal, otherwise every window
# that crosses 2025-08-31 would be lost.
daily_fill = daily.copy()
for c in daily_fill.columns:
    first = daily_fill[c].first_valid_index()
    seg = daily_fill.loc[first:JAE_END, c]
    daily_fill.loc[first:JAE_END, c] = seg.fillna(0)
variants, variants_live = {}, {}
for w in MA_SEQ:
    m = daily.rolling(w).mean().resample('ME').mean()
    m_live = daily_fill.rolling(w).mean().resample('ME').mean()
    for c in daily.columns:
        variants[f'{c}_ma{w}'] = m[c]
        variants_live[f'{c}_ma{w}'] = m_live[c]
variants = pd.DataFrame(variants)
variants_live = pd.DataFrame(variants_live)
variants.loc[variants.index > JAE_END] = variants_live.loc[variants_live.index > JAE_END]
variants = variants[variants.index <= last_day + pd.offsets.MonthEnd(0)]

# check: the JAE months must reproduce data/df_jae_reddit.xlsx
jae = pd.read_excel(os.path.join(REPLICATION_DIR, 'data', 'df_jae_reddit.xlsx'), index_col=0, parse_dates=True)
chk = [np.nanmax(np.abs(variants.loc[:JAE_END, c].values - jae.loc[:JAE_END, c].reindex(variants.loc[:JAE_END].index).values))
       for c in variants.columns if c in jae.columns]
print(f'replication check vs df_jae_reddit.xlsx: max abs diff {max(chk):.2e} over {len(chk)} series')

# ----------------------------------------------------------------------------
# 6b. Fine-tuned classifiers (labels from label_finetuned.py), same aggregation
#     as build_daily_signals.build_finetuned: thread = sign of the mean trend of
#     its items, dated at the submission, summed by day; outage days from the
#     Factiva titles labelled by the same model. A model enters only once every
#     live item has its label.
# ----------------------------------------------------------------------------
ft_models, ft_daily_live = [], pd.DataFrame(index=live_days)
for mname in FT_MODELS:
    fpath = os.path.join(FT_DIR, f'{mname}.csv')
    if MOCK or not os.path.exists(fpath):
        continue
    fl = pd.read_csv(fpath, dtype={'id': str})
    it = items_out.merge(fl[['id', 'kind', 'trend']], on=['id', 'kind'], how='left')
    if it['trend'].isna().any():
        print(f'fine-tuned {mname}: {int(it["trend"].isna().sum())} items not labelled yet - left out')
        continue
    rd_ = it[it['kind'].isin(['submission', 'comment'])]
    rd_ = rd_[rd_['id_sub'].isin(set(rd_.loc[rd_['kind'] == 'submission', 'id_sub']))]
    th = rd_.groupby('id_sub').agg(date=('created', 'min'), subreddit=('subreddit', 'first'), value=('trend', 'mean'))
    th['date'] = pd.to_datetime(th['date']).dt.floor('D')
    th['disc'] = np.where(th['value'] > 0, 1, np.where(th['value'] < 0, -1, 0))
    fday = None
    if factiva_days:
        fv = it[it['kind'] == 'factiva']
        fday = fv['trend'].groupby(pd.to_datetime(fv['created']).dt.floor('D')).sum().reindex(factiva_days, fill_value=0).astype(float)
    for s_, suf in FT_SUFFIX.items():
        col = f'{mname}_trend_{suf}'
        ft_daily_live[col] = th[th['subreddit'] == s_].groupby('date')['disc'].sum().reindex(live_days, fill_value=0).astype(float)
        ft_daily_live.loc[missing_days, col] = np.nan
        if fday is not None:
            ft_daily_live.loc[factiva_days, col] = fday.values
    ft_models.append(mname)

ft_variants = None
if ft_models:
    hist_ft = pd.read_csv(os.path.join(REPLICATION_DIR, 'data', 'daily_signals_finetuned.csv'), index_col=0, parse_dates=True)
    cols_ft = [f'{m}_trend_{suf}' for m in ft_models for suf in FT_SUFFIX.values()]
    hist_ft = hist_ft.loc[hist_ft.index <= JAE_END, cols_ft]
    dft = pd.concat([hist_ft, ft_daily_live[cols_ft]], axis=0)
    dft = dft[~dft.index.duplicated(keep='first')].asfreq('D')
    dft_fill = dft.copy()
    for c in dft_fill.columns:
        first = dft_fill[c].first_valid_index()
        dft_fill.loc[first:JAE_END, c] = dft_fill.loc[first:JAE_END, c].fillna(0)
    fv_, fv_live = {}, {}
    for w in MA_SEQ:
        m1 = dft.rolling(w).mean().resample('ME').mean()
        m2 = dft_fill.rolling(w).mean().resample('ME').mean()
        for c in cols_ft:
            fv_[f'{c}_ma{w}'] = m1[c]
            fv_live[f'{c}_ma{w}'] = m2[c]
    ft_variants = pd.DataFrame(fv_)
    fv_live = pd.DataFrame(fv_live)
    ft_variants.loc[ft_variants.index > JAE_END] = fv_live.loc[fv_live.index > JAE_END]
    ft_variants = ft_variants[ft_variants.index <= last_day + pd.offsets.MonthEnd(0)]
    chk = [np.nanmax(np.abs(ft_variants.loc[:JAE_END, c].values - jae.loc[:JAE_END, c].reindex(ft_variants.loc[:JAE_END].index).values))
           for c in ft_variants.columns if c in jae.columns]
    print(f'fine-tuned: {len(ft_models)} models; replication check vs df_jae_reddit.xlsx: max abs diff {max(chk):.2e} over {len(chk)} series')

# ----------------------------------------------------------------------------
# 7. CPI and core PCE (FRED, latest vintage), 12-month log change as in the paper
#    order: FRED API (key) -> keyless fredgraph.csv -> last successful download
# ----------------------------------------------------------------------------
levels, macro_source = {}, {}
for sid in ['CPIAUCSL', 'PCEPILFE']:
    s = None
    if FRED_API_KEY:
        try:
            r = requests.get('https://api.stlouisfed.org/fred/series/observations',
                             params={'series_id': sid, 'api_key': FRED_API_KEY, 'file_type': 'json', 'observation_start': '2001-01-01'}, timeout=90)
            r.raise_for_status()
            obs = pd.DataFrame(r.json()['observations'])
            s = pd.Series(pd.to_numeric(obs['value'], errors='coerce').values, index=pd.to_datetime(obs['date']))
            macro_source[sid] = 'FRED API'
        except Exception as e:
            print(f'FRED API {sid}: {type(e).__name__}')
    if s is None:
        try:
            r = requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}', timeout=90, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            obs = pd.read_csv(StringIO(r.text))
            s = pd.Series(pd.to_numeric(obs[sid], errors='coerce').values, index=pd.to_datetime(obs[obs.columns[0]]))
            macro_source[sid] = 'FRED csv'
        except Exception as e:
            print(f'FRED csv {sid}: {type(e).__name__}')
    if s is not None:
        levels[sid] = s.dropna().resample('ME').mean()
if len(levels) == 2:
    pd.DataFrame(levels).to_csv(MACRO_CACHE)
else:
    if os.path.exists(MACRO_CACHE):
        cached = pd.read_csv(MACRO_CACHE, index_col=0, parse_dates=True)
        for sid in ['CPIAUCSL', 'PCEPILFE']:
            if sid not in levels:
                levels[sid] = cached[sid].dropna().resample('ME').mean(); macro_source[sid] = 'cache'
    for sid, col in [('CPIAUCSL', 'infl_cpi'), ('PCEPILFE', 'infl_pce')]:
        if sid not in levels:
            levels[sid] = jae[col].dropna().resample('ME').mean(); macro_source[sid] = 'JAE frozen (2025M8)'
    print(f'WARNING: FRED not reachable, macro from {macro_source}')
macro = pd.DataFrame({sid: 100 * np.log(levels[sid]).diff(12) for sid in ['CPIAUCSL', 'PCEPILFE']})
print('macro through:', {k: str(macro[k].dropna().index.max().date()) for k in macro}, macro_source)

# ----------------------------------------------------------------------------
# 8. Chart series: z-scores on the fixed JAE window, as in the paper's chart
#    (RIS: median and 10-90% band across the 24 Llama-70B variants)
# ----------------------------------------------------------------------------
months = pd.date_range(CHART_START, max(variants.index.max(), macro.dropna(how='all').index.max()), freq='ME')
V = variants.reindex(months)
Vz = (V - variants.loc[ZSCORE_START:ZSCORE_END].mean()) / variants.loc[ZSCORE_START:ZSCORE_END].std()
n_avail = Vz.notna().sum(axis=1)
ris_med = Vz.median(axis=1).where(n_avail >= 3)
ris_p10 = Vz.quantile(0.10, axis=1).where(n_avail >= 3)
ris_p90 = Vz.quantile(0.90, axis=1).where(n_avail >= 3)
ris_ft = None
if ft_variants is not None:
    F = ft_variants.reindex(months)
    Fz = (F - ft_variants.loc[ZSCORE_START:ZSCORE_END].mean()) / ft_variants.loc[ZSCORE_START:ZSCORE_END].std()
    nf = Fz.notna().sum(axis=1)
    ris_ft = {'median': Fz.median(axis=1).where(nf >= 3), 'p10': Fz.quantile(0.10, axis=1).where(nf >= 3),
              'p90': Fz.quantile(0.90, axis=1).where(nf >= 3), 'n_variants': nf}
M = macro.reindex(months)
Mz = (M - macro.loc[ZSCORE_START:ZSCORE_END].mean()) / macro.loc[ZSCORE_START:ZSCORE_END].std()

rd = lambda x: [None if pd.isna(v) else round(float(v), 3) for v in x]
gap_months = sorted({d.strftime('%Y-%m') for d in still_missing})
factiva_months = sorted({d.strftime('%Y-%m') for d in factiva_days})
out = {
    'meta': {'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'mock': MOCK, 'model': MODEL,
             'live_start': LIVE_START.strftime('%Y-%m'), 'data_through': str(last_day.date()),
             'reddit_through': data_through, 'gap_months': gap_months,
             'factiva_days': len(factiva_days), 'factiva_months': factiva_months, 'macro_source': macro_source,
             'macro_last': {k: macro[k].dropna().index.max().strftime('%Y-%m') for k in macro},
             'n_live_labelled': {'submission': int((lab['kind'] == 'submission').sum()), 'comment': int((lab['kind'] == 'comment').sum())},
             'n_pending': n_pending, 'zscore_window': [ZSCORE_START[:7], ZSCORE_END[:7]]},
    'months': [d.strftime('%Y-%m') for d in months],
    'ris': {'median': rd(ris_med), 'p10': rd(ris_p10), 'p90': rd(ris_p90), 'n_variants': [int(x) for x in n_avail]},
    'cpi': {'z': rd(Mz['CPIAUCSL']), 'pct': rd(M['CPIAUCSL'])},
    'pce': {'z': rd(Mz['PCEPILFE']), 'pct': rd(M['PCEPILFE'])},
    'ris_ft': None if ris_ft is None else {'median': rd(ris_ft['median']), 'p10': rd(ris_ft['p10']), 'p90': rd(ris_ft['p90']),
                                           'n_variants': [int(x) for x in ris_ft['n_variants']], 'models': ft_models},
}
with open(OUT_FILE, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'wrote {OUT_FILE} - {len(months)} months, live labelled {out["meta"]["n_live_labelled"]}, pending {n_pending}')

# ----------------------------------------------------------------------------
# 9. Publish
# ----------------------------------------------------------------------------
if PUSH and not MOCK:
    subprocess.run(['git', '-C', REPO_DIR, 'add', 'docs/data/dashboard.json', 'data/daily_signals_llama70b_live.csv', 'data/macro_fred.csv', 'labels'], check=True)
    if subprocess.run(['git', '-C', REPO_DIR, 'diff', '--cached', '--quiet']).returncode != 0:
        subprocess.run(['git', '-C', REPO_DIR, 'commit', '-m', f'RIS update {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC'], check=True)
        subprocess.run(['git', '-C', REPO_DIR, 'push'], check=True)
        print('pushed')
    else:
        print('nothing new to push')
print(f'done in {time.time() - t0:.0f}s')
