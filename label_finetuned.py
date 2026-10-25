# =============================================================================
# Fine-tuned classifiers of the paper (Hugging Face collection
# andreadm/reddit-infla-pulse) on the Mac, no GPU.
#
#   VALIDATE=1 python label_finetuned.py   check against the published test metrics
#   python label_finetuned.py              label the live posts (cache in labels/finetuned/)
#
# Decoder models (Gemma-2 2B, Llama-3.2 1B/3B, Qwen2.5 0.5B/1.5B, xQDoRA+) are
# PEFT adapters trained on a 4-bit NF4 base. 4-bit loading needs CUDA, so here
# the base is loaded in bfloat16 (the compute dtype of training) and the
# adapter is attached on top: an approximation, measured by VALIDATE=1.
# InflaBERT is a full checkpoint and loads exactly.
#
# Inference as in andrea-dm/reddit-pulse (src/reddit/inference): raw text
# (submission title / comment body), left padding, truncation at 1024 tokens
# (512 for BERT), argmax over {0: down, 1: neutral, 2: up} -> {-1, 0, 1}.
# =============================================================================
import os, sys, json, time, gc
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DESK = os.path.expanduser('~/Desktop')
HF_TOKEN_FILE = os.environ.get('HF_TOKEN_FILE', os.path.join(HOME_DESK, 'subreddits', 'hf_token.txt'))
GOLD_FILE = os.environ.get('RIS_GOLD_FILE', os.path.join(HOME_DESK, 'subreddits', 'labelled_gold.xlsx'))
ITEMS_FILE = os.path.join(REPO_DIR, 'data', 'items_live.csv')                # written by update_dashboard.py (not committed)
OUT_DIR = os.path.join(REPO_DIR, 'labels', 'finetuned')
VALIDATE = os.environ.get('VALIDATE', '0') == '1'
ONLY = [m for m in os.environ.get('MODELS', '').split(',') if m]
BATCH = int(os.environ.get('BATCH', '8'))
CHECKPOINT_EVERY = 20                                                        # batches

# (column name in the paper's files, Hub repo, kind)
MODELS = [
    ('qwen2_5_0_5b_xqdora', 'andreadm/reddit-pulse-qwen2.5_0.5b-xqdora', 'decoder'),
    ('qwen2_5_1_5b_xqdora', 'andreadm/reddit-pulse-qwen2.5_1.5b-xqdora', 'decoder'),
    ('llama3_2_1b_xqdora', 'andreadm/reddit-pulse-llama3.2_1b-xqdora', 'decoder'),
    ('llama3_2_3b_xqdora', 'andreadm/reddit-pulse-llama3.2_3b-xqdora', 'decoder'),
    ('gemma2_2b_xqdora', 'andreadm/reddit-pulse-gemma2_2b-xqdora', 'decoder'),
    ('inflabert', 'andreadm/reddit-pulse-bert', 'bert'),
]
if ONLY:
    MODELS = [m for m in MODELS if m[0] in ONLY]
ENCODING = {'down': -1, 'neutral': 0, 'up': 1}
LABEL2ID = {'down': 0, 'neutral': 1, 'up': 2}

if os.path.exists(HF_TOKEN_FILE):
    os.environ['HF_TOKEN'] = open(HF_TOKEN_FILE).read().strip()
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast
from peft import PeftModel

DEVICE = os.environ.get('RIS_DEVICE', 'cpu')                                # fixed Mac: MPS on its AMD GPU gives wrong outputs (all DOWN); 'mps' to force it
DTYPE = torch.bfloat16 if DEVICE != 'cpu' else torch.float32             # bf16 on this Xeon CPU is emulated and slow
try:
    torch.zeros(2, dtype=DTYPE, device=DEVICE) @ torch.zeros(2, dtype=DTYPE, device=DEVICE)
except Exception:
    DTYPE = torch.float32
print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] device {DEVICE}, dtype {DTYPE}, torch {torch.__version__}')
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# What to label
# ----------------------------------------------------------------------------
if VALIDATE:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score
    gold = pd.read_excel(GOLD_FILE)
    gold = gold.dropna(subset=['title', 'Label'])[['title', 'Label']].copy()          # = reddit.data.preparation
    gold['label_str'] = gold['Label'].str.lower()
    gold['label'] = gold['label_str'].map(LABEL2ID)
    gold = gold.rename(columns={'title': 'text'})
    gold['id'] = np.arange(len(gold)).astype(str)
    gold['kind'] = 'gold'
    items = gold[['id', 'kind', 'text']]
    print(f'gold set: {len(gold)} titles')
    report = []
else:
    items = pd.read_csv(ITEMS_FILE, dtype={'id': str})
    items = items.dropna(subset=['text'])
    print(f'live items: {len(items)} ({items["kind"].value_counts().to_dict()})')

# ----------------------------------------------------------------------------
# Loop over models
# ----------------------------------------------------------------------------
for name, repo, kind in MODELS:
    t0 = time.time()
    out_file = os.path.join(OUT_DIR, f'{name}.csv')
    if VALIDATE:
        todo = items
        done = pd.DataFrame(columns=['id', 'kind', 'trend'])
    else:
        done = pd.read_csv(out_file, dtype={'id': str}) if os.path.exists(out_file) else pd.DataFrame(columns=['id', 'kind', 'trend'])
        keys = set(done['id'].astype(str) + '|' + done['kind'].astype(str))
        todo = items[~(items['id'] + '|' + items['kind']).isin(keys)]
    print(f'\n== {name}: {len(todo)} to label')
    if len(todo) == 0:
        continue

    # -- load ---------------------------------------------------------------
    try:                                                                     # tokenizer files saved by transformers 5: extra_special_tokens as a list (4.x wants a dict;
        tok = AutoTokenizer.from_pretrained(repo, use_fast=True, extra_special_tokens={})   # the tokens are already in tokenizer.json)
    except ValueError as e:                                                  # class 'TokenizersBackend' (5.x name): same tokenizer.json, generic fast class
        if 'TokenizersBackend' not in str(e):
            raise
        tok = PreTrainedTokenizerFast.from_pretrained(repo, extra_special_tokens={})
    if kind == 'decoder':
        tok.pad_token = tok.pad_token or tok.eos_token
        tok.padding_side = 'left'
        adapter_cfg = json.load(open(hf_hub_download(repo, 'adapter_config.json')))
        base_id = adapter_cfg['base_model_name_or_path']
        cfg = AutoConfig.from_pretrained(repo, num_labels=3, id2label={0: 'down', 1: 'neutral', 2: 'up'},
                                         label2id=LABEL2ID, use_cache=False)
        for a in ('quantization_config', '_pre_quantization_dtype'):
            if a in cfg.__dict__:
                delattr(cfg, a)
        base = AutoModelForSequenceClassification.from_pretrained(base_id, config=cfg, dtype=DTYPE)
        model = PeftModel.from_pretrained(base, repo)
        model.config.pad_token_id = tok.pad_token_id
        max_len = 1024
    else:
        model = AutoModelForSequenceClassification.from_pretrained(repo, dtype=torch.float32)
        max_len = 512
    dev = DEVICE
    try:
        model.to(dev).eval()
    except RuntimeError as e:                                                # GPU memory too small for this model (AMD, ~5 GB): use the CPU
        if dev != 'mps' or 'out of memory' not in str(e):
            raise
        model.to('cpu').eval()
        torch.mps.empty_cache()
        dev = 'cpu'
        print(f'  MPS out of memory - {name} runs on the CPU', flush=True)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    print(f'  loaded {repo} ({"base " + base_id if kind == "decoder" else "full checkpoint"}), id2label {id2label}, {time.time() - t0:.0f}s')

    # -- predict (shortest first, so batches pad little) --------------------
    order = todo.assign(_n=todo['text'].str.len()).sort_values('_n')
    rows, nb = [], 0
    for i in range(0, len(order), BATCH):
        chunk = order.iloc[i:i + BATCH]
        enc = tok(chunk['text'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        with torch.inference_mode():
            logits = model(input_ids=enc['input_ids'].to(dev), attention_mask=enc['attention_mask'].to(dev)).logits
        pred = torch.argmax(logits.float(), dim=1).cpu().numpy()
        for (rid, rkind), p in zip(chunk[['id', 'kind']].itertuples(index=False), pred):
            rows.append({'id': rid, 'kind': rkind, 'trend': ENCODING.get(id2label.get(int(p), ''), np.nan)})
        nb += 1
        if not VALIDATE and nb % CHECKPOINT_EVERY == 0:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(out_file, index=False)
            print(f'  {len(rows)}/{len(order)}  ({time.time() - t0:.0f}s)', flush=True)
    res = pd.DataFrame(rows)

    if VALIDATE:
        g = gold.merge(res, on=['id', 'kind'])
        g['pred'] = g['trend'].map({-1: 0, 0: 1, 1: 2})
        # rebuild the test split of the published (median) seed
        mt = pd.read_csv(hf_hub_download(repo, 'evaluation/seeds_test_metrics.csv'))
        fcol = 'f1_weighted'
        sel = mt[mt['selected'].astype(str).str.lower() == 'true']
        med = sel.iloc[0] if len(sel) else mt.sort_values(fcol).reset_index(drop=True).iloc[len(mt) // 2]
        seed = int(med['seed'])
        _, test = train_test_split(g, test_size=0.10, stratify=g['label'], random_state=seed)
        f1_test = f1_score(test['label'], test['pred'], average='weighted')
        f1_all = f1_score(g['label'], g['pred'], average='weighted')
        r = {'model': name, 'seed': seed, 'published_test_f1': round(float(med[fcol]), 4), 'mac_test_f1': round(f1_test, 4),
             'mac_full_gold_f1': round(f1_all, 4), 'mac_full_gold_acc': round(accuracy_score(g['label'], g['pred']), 4),
             'secs': round(time.time() - t0)}
        report.append(r)
        print('  ', r, flush=True)
        pd.DataFrame(report).to_csv(os.path.join(OUT_DIR, 'validation_report.csv'), index=False)
    else:
        pd.concat([done, res], ignore_index=True).to_csv(out_file, index=False)
        print(f'  done: {len(res)} labelled in {time.time() - t0:.0f}s, trend counts {res["trend"].value_counts().to_dict()}')

    del model
    if kind == 'decoder':
        del base
    gc.collect()
    if DEVICE == 'mps':
        torch.mps.empty_cache()

if VALIDATE:
    print('\nVALIDATION (weighted F1; published = 4-bit on GPU, mac = bf16 base + adapter)')
    print(pd.DataFrame(report).to_string(index=False))
print(f'\nfinished {datetime.now():%H:%M:%S}')
