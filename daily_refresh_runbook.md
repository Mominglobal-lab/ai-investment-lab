# Daily Refresh Runbook

## Daily Command

Run this once per day from the project root to refresh everything:

```powershell
python scripts\run_scheduled_refresh.py --benchmark SPY
```

This is the full refresh command. It refreshes:

- stock caches for `S&P 500` and `Nasdaq 100`
- fixed-income caches for `US Treasuries` and `Bond ETFs`
- treasury-yield cache
- prices cache
- model artifacts
- explainability artifacts
- uncertainty artifacts
- monitoring artifacts

## Recommended Environment

From the repo root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_scheduled_refresh.py --benchmark SPY
```

## Windows Task Scheduler Settings

Use these values in Task Scheduler.

Program:

```text
C:\Users\bhuiyana\Documents\GitHub\ai-investment-lab\.venv\Scripts\python.exe
```

Arguments:

```text
scripts\run_scheduled_refresh.py --benchmark SPY
```

Start in:

```text
C:\Users\bhuiyana\Documents\GitHub\ai-investment-lab
```

## Notes

- The Streamlit UI is now cache-first and does not expose manual refresh buttons.
- If the scheduled job does not run, users will see stale or missing cache data in the app.
- If needed, you can skip individual stages with flags such as `--skip-monitoring` or `--skip-treasury`.
