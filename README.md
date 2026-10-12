# Reddit Inflation Signal (RIS), live

Live extension of the Llama-70B Reddit Inflation Signal in Del Monaco, Longo, Marcucci and Tafani, *Reddit's Pulse on US Inflation: Forecasting with Large Language Models*. The paper sample ends in 2025M8; this repository labels everything from **1 September 2025** onward and publishes it on GitHub Pages.

## Pipeline (`update_dashboard.py`)

1. Reads `{Economics,economy,wallstreetbets}_inflation_{submissions,comments}.csv` from `~/Desktop/subreddits`, posts from **1 September 2025** on.
2. Keyword screen of the paper: inflation, deflation, disinflation, hyperinflation, price, prices (r/wallstreetbets without price/prices).
3. Geographic screen of submissions and UP / DOWN / NEUTRAL labels of submissions and comments with `llama-3.3-70b-instruct` on GPT@JRC (OpenAI client, temperature 0, prompts of the Online Appendix). Comments enter if their submission is kept and they are posted within 14 days.
4. Daily signal per subreddit with the aggregation of `reddit_pulse_replication/_build/build_daily_signals.py` (`build_llama70`), appended to `reddit_pulse_replication/data/daily_signals_llama70b.csv`.
5. Monthly variants as `monthly_merge` in `forecast_codes/_common.py`: `daily.rolling(w).mean().resample('M').mean()`, w in 1, 5, 10, 30, 90, 120, 180, 360. Each run checks that the months up to 2025M8 reproduce `data/df_jae_reddit.xlsx`.
6. CPIAUCSL and PCEPILFE from FRED, 12-month log change. RIS (median and 10-90% band of the 24 variants) and inflation shown as z-scores over 2009M1-2025M8.
7. Writes `docs/data/dashboard.json`; with `PUSH=1` commits and pushes.

Labels are cached in `labels/`, so each run only calls the model on new posts; an interrupted run resumes.

## Setup on the Mac

```bash
cd ~/Desktop/subreddits/reddit-inflation-pulse
python3 update_dashboard.py          # first run: ~2,400 geo + ~13,000 signal calls, then incremental
git init -b main && git add . && git commit -m "RIS live dashboard"
gh repo create reddit-inflation-signal --public --source=. --remote=origin --push
# GitHub: Settings -> Pages -> Deploy from branch: main, folder /docs
./install_launchd.sh                 # hourly updates
```

Paths (environment variables or `.env`): `RIS_REDDIT_DIR` (default `~/Desktop/subreddits`), `RIS_REPLICATION_DIR` (`~/Desktop/reddit_pulse_replication`), `GPTJRC_TOKEN_FILE` (`~/Desktop/live_project/tokens/token-key.jwt`), `FRED_API_KEY` (optional; without it the keyless fredgraph.csv endpoint is used).

`MOCK_LLM=1` runs everything offline with a keyword stand-in, writing to `labels_mock/` and `docs/data/dashboard_mock.json` (open the page with `?data=dashboard_mock.json`).

## Notes

- Days with no rows at all in the source CSVs are set to missing, not to zero signal.
- Comments longer than 6,000 characters are truncated before labeling.
