# Daily Gemini weak-labeling trickle (Windows Task Scheduler entry point).
# Free tier is 20 requests/day for gemini-3.6-flash (confirmed empirically
# 2026-08-12, NOT the ~1500/day figure some docs report) - at batch_size=5
# that's ~100 rows/day. The script now stops itself cleanly once the daily
# quota 429s (see GeminiDailyQuotaExhausted in src/weak_labeling/llm_extract.py)
# rather than retrying uselessly through every remaining batch, so this is
# safe to just let run to completion (or quota exhaustion) unattended.
#
# Registered as a scheduled task - see docs/baseline_plan.md "Step 2" for
# the exact schtasks command used to set it up.

$ErrorActionPreference = "Continue"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ProjectDir = "C:\Users\Brij Nandan Dogra\Desktop\RSNA Knee Abnormality Detection"
$LogDir = Join-Path $ProjectDir "outputs\weak_labeling\daily_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$LogFile = Join-Path $LogDir "gemini_daily_$Timestamp.log"

Set-Location $ProjectDir
& python scripts\weak_label_reports.py --mode full --provider gemini --batch-size 5 --workers 1 *>&1 |
    Tee-Object -FilePath $LogFile
