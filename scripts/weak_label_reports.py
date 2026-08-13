"""Weak-label knee MRI reports via LLM extraction (src/weak_labeling/llm_extract.py).

Three providers (--provider):
  claude - shells out to the authenticated Claude Code CLI. Used for the
           initial 58-row validation and the first 300 rows of the full run.
  groq   - calls Groq's API directly (GROQ_API_KEY from env or .env), to
           avoid burning Claude Code session credits on the bulk of the
           run. Also emits a per-report "confidence" (high/low) field.
  gemini - calls the plain Gemini Developer API directly (GEMINI_API_KEY
           from env or .env; free tier, NOT Vertex AI/Agent Platform -
           that route was abandoned after an unresolved billing block).
           Also emits a per-report "confidence" field.

Two modes:
  validate - run extraction on the 58 rows that already have real structured
             labels (withholding those labels from the model), compare
             extracted vs. true, report precision/recall/agreement per
             target. Run this and check the numbers BEFORE trusting `full`.
             With --provider groq or gemini, also reports whether
             low-confidence rows correlate with actual errors.
  full     - run extraction on the 4349 report-only rows, write
             data/processed/weak_labels.csv (all rows) plus
             data/processed/weak_labels_high_confidence.csv /
             _low_confidence.csv (confidence-based split for providers that
             emit one; rows from the claude provider have no confidence
             signal and are treated as high-confidence, since that phase
             was separately validated).

Checkpointed per provider (outputs/weak_labeling/weak_extracted.csv for
claude, weak_extracted_groq.csv / weak_extracted_gemini.csv for the others)
so progress survives interruptions. Resuming skips rows already done by ANY
provider's checkpoint, not just the current one - a rerun never reprocesses
rows another provider already labeled.

Usage:
    python scripts/weak_label_reports.py --mode validate --provider gemini
    python scripts/weak_label_reports.py --mode full --provider gemini --workers 1
    python scripts/weak_label_reports.py --mode full --provider gemini --limit 100   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.weak_labeling.llm_extract import (
    GEMINI_MODEL_DEFAULT,
    GROQ_MODEL_DEFAULT,
    TARGETS,
    GeminiDailyQuotaExhausted,
    extract_batch,
)

ID_COL = "StudyInstanceUID"
DEFAULT_MODELS = {"claude": "claude-sonnet-5", "groq": GROQ_MODEL_DEFAULT, "gemini": GEMINI_MODEL_DEFAULT}


def batched(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _run_one_batch(i: int, batch: list, model: str, provider: str) -> tuple[int, list, float, Exception | None]:
    t_batch = time.time()
    try:
        result = extract_batch(batch, model=model, provider=provider)
    except Exception as e:  # noqa: BLE001 - keep going, this batch just won't have weak labels
        return i, [], time.time() - t_batch, e
    rows = []
    for report_id, _ in batch:
        row = {ID_COL: report_id}
        row.update(result.get(report_id, {t: None for t in TARGETS}))
        rows.append(row)
    return i, rows, time.time() - t_batch, None


def checkpoint_path_for(out_dir: Path, provider: str, prefix: str = "weak_extracted") -> Path:
    # "claude" keeps the original filename (no suffix) for backward
    # compatibility with the checkpoint already on disk from before --provider existed.
    suffix = "" if provider == "claude" else f"_{provider}"
    return out_dir / f"{prefix}{suffix}.csv"


def other_provider_done_ids(out_dir: Path, current_checkpoint: Path, prefix: str = "weak_extracted") -> set[str]:
    """IDs already extracted by ANY provider's checkpoint other than the one
    we're about to write to - so switching providers mid-run never reprocesses
    rows another provider already labeled."""
    ids: set[str] = set()
    if not out_dir.exists():
        return ids
    for p in out_dir.glob(f"{prefix}*.csv"):
        if p == current_checkpoint or not p.exists():
            continue
        try:
            ids |= set(pd.read_csv(p)[ID_COL])
        except (pd.errors.EmptyDataError, KeyError):
            continue
    return ids


def run_extraction(df: pd.DataFrame, batch_size: int, model: str, checkpoint_path: Path,
                    workers: int, provider: str, extra_done_ids: set[str] | None = None) -> pd.DataFrame:
    """Extract weak labels for every row in df, resuming from checkpoint_path
    (plus extra_done_ids from other providers' checkpoints) for IDs already done."""
    done_ids: set[str] = set(extra_done_ids or set())
    if checkpoint_path.exists():
        prior = pd.read_csv(checkpoint_path)
        done_ids |= set(prior[ID_COL])
    if done_ids:
        print(f"Resuming: {len(done_ids)} rows already extracted (this + other providers' checkpoints)")

    remaining = df[~df[ID_COL].isin(done_ids)]
    id_text_pairs = list(zip(remaining[ID_COL], remaining["Report"]))
    batches = list(batched(id_text_pairs, batch_size))
    print(f"{len(remaining)} rows to extract in {len(batches)} batches of up to {batch_size}, "
          f"{workers} concurrent worker(s), provider={provider}, model={model}")

    write_header = not checkpoint_path.exists()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    t0 = time.time()
    n_done = 0
    quota_exhausted = False
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(_run_one_batch, i, b, model, provider) for i, b in enumerate(batches, 1)]
    try:
        for future in as_completed(futures):
            i, rows, elapsed, err = future.result()
            n_done += 1
            if isinstance(err, GeminiDailyQuotaExhausted):
                # Retrying/skipping through every remaining batch would just
                # burn time - a per-day quota won't reset within this run.
                print(f"  batch {i}/{len(batches)}: Gemini daily quota exhausted - stopping here "
                      f"({len(batches) - n_done} batches not attempted). Re-run later (e.g. the next "
                      f"scheduled daily run) once the quota resets to continue.")
                quota_exhausted = True
                break
            if err is not None:
                print(f"  batch {i}/{len(batches)} FAILED ({err}); skipping")
                continue
            with write_lock:
                batch_df = pd.DataFrame(rows)
                batch_df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)
                write_header = False
            print(f"  batch {i}/{len(batches)} done ({len(rows)} rows, {elapsed:.1f}s)  "
                  f"[{n_done}/{len(batches)} batches, total elapsed {time.time() - t0:.1f}s]")
    finally:
        pool.shutdown(wait=not quota_exhausted, cancel_futures=quota_exhausted)

    return pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame(columns=[ID_COL] + TARGETS)


def validate(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.labels_csv)
    labeled = df[df[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    print(f"Validating against {len(labeled)} rows with real structured labels (provider={args.provider})")

    out_dir = Path(args.out_dir)
    checkpoint = checkpoint_path_for(out_dir, args.provider, prefix="validation_extracted")
    extracted = run_extraction(labeled, args.batch_size, args.model, checkpoint, args.workers, args.provider)

    merged = labeled.merge(extracted, on=ID_COL, suffixes=("_true", "_pred"))

    print("\n=== Per-target validation metrics ===")
    print(f"{'target':20s} {'n':>4s} {'agree':>7s} {'precision':>10s} {'recall':>8s} {'tp':>4s} {'fp':>4s} {'fn':>4s} {'tn':>4s} {'null':>5s}")
    summary_rows = []
    for t in TARGETS:
        true_col, pred_col = f"{t}_true", f"{t}_pred"
        sub = merged[[true_col, pred_col]].dropna(subset=[true_col])
        n = len(sub)
        n_null_pred = sub[pred_col].isna().sum()
        scored = sub.dropna(subset=[pred_col])
        tp = ((scored[true_col] == 1) & (scored[pred_col] == 1)).sum()
        fp = ((scored[true_col] == 0) & (scored[pred_col] == 1)).sum()
        fn = ((scored[true_col] == 1) & (scored[pred_col] == 0)).sum()
        tn = ((scored[true_col] == 0) & (scored[pred_col] == 0)).sum()
        agree = (tp + tn) / n if n else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"{t:20s} {n:4d} {agree:7.1%} {precision:10.1%} {recall:8.1%} {tp:4d} {fp:4d} {fn:4d} {tn:4d} {n_null_pred:5d}")
        summary_rows.append({"target": t, "n": n, "agreement": agree, "precision": precision,
                              "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "null_pred": n_null_pred})

    summary_df = pd.DataFrame(summary_rows)
    overall_agree = summary_df["agreement"].mean()
    print(f"\nMean per-target agreement: {overall_agree:.1%}")

    if args.provider in ("groq", "gemini") and "confidence" in merged.columns:
        print("\n=== Confidence vs. actual error rate ===")
        rows_err = []
        for _, row in merged.iterrows():
            n_scored, n_wrong = 0, 0
            for t in TARGETS:
                tv, pv = row.get(f"{t}_true"), row.get(f"{t}_pred")
                if pd.isna(tv) or pd.isna(pv):
                    continue
                n_scored += 1
                n_wrong += int(tv != pv)
            rows_err.append({"confidence": row.get("confidence"), "n_scored": n_scored, "n_wrong": n_wrong,
                              "error_rate": n_wrong / n_scored if n_scored else float("nan")})
        err_df = pd.DataFrame(rows_err)
        for conf_val, grp in err_df.groupby("confidence", dropna=False):
            print(f"  confidence={conf_val!r}: n_rows={len(grp)}  mean per-row error rate={grp['error_rate'].mean():.1%}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"validation_summary_{args.provider}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    merged.to_csv(out_dir / f"validation_merged_{args.provider}.csv", index=False)


def full_run(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.labels_csv)
    unlabeled = df[df[TARGETS].isna().all(axis=1)].reset_index(drop=True)
    if args.limit:
        unlabeled = unlabeled.head(args.limit)
    print(f"Weak-labeling {len(unlabeled)} report-only rows (provider={args.provider})")

    out_dir = Path(args.out_dir)
    checkpoint = checkpoint_path_for(out_dir, args.provider)
    prior_ids = other_provider_done_ids(out_dir, checkpoint)
    run_extraction(unlabeled, args.batch_size, args.model, checkpoint, args.workers, args.provider, prior_ids)

    # Final output combines every provider's checkpoint, not just this run's,
    # tagging each row with which provider actually labeled it (parsed from
    # the checkpoint filename: weak_extracted.csv -> claude,
    # weak_extracted_<provider>.csv -> <provider>).
    all_checkpoints = list(out_dir.glob("weak_extracted*.csv"))
    parts = []
    for p in all_checkpoints:
        stem_suffix = p.stem[len("weak_extracted"):]  # "" or "_groq" / "_gemini"
        provider_name = stem_suffix.lstrip("_") or "claude"
        df_p = pd.read_csv(p)
        df_p["label_source"] = f"weak_{provider_name}"
        parts.append(df_p)
    extracted = pd.concat(parts, ignore_index=True)
    extracted = extracted.drop_duplicates(subset=[ID_COL], keep="last")

    out = unlabeled[[ID_COL, "Report"]].merge(extracted, on=ID_COL)
    if "confidence" not in out.columns:
        out["confidence"] = None
    out = out[[ID_COL, "label_source", "confidence"] + TARGETS + ["Report"]]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={out.shape}")

    # Bucket: rows with no confidence signal (claude-provider rows, already
    # separately validated) count as high-confidence by default.
    is_low = out["confidence"] == "low"
    high_conf = out[~is_low]
    low_conf = out[is_low]
    high_path = out_path.parent / "weak_labels_high_confidence.csv"
    low_path = out_path.parent / "weak_labels_low_confidence.csv"
    high_conf.to_csv(high_path, index=False)
    low_conf.to_csv(low_path, index=False)
    print(f"Wrote {high_path}  shape={high_conf.shape}")
    print(f"Wrote {low_path}  shape={low_conf.shape}")

    n_any = out[TARGETS].notna().any(axis=1).sum()
    n_all_null = (out[TARGETS].isna().all(axis=1)).sum()
    print(f"Rows with >=1 non-null weak label: {n_any} / {len(out)}")
    print(f"Rows with ALL targets null (extraction failed/unusable): {n_all_null} / {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "full"], required=True)
    parser.add_argument("--provider", choices=["claude", "groq", "gemini"], default="claude")
    parser.add_argument("--labels-csv", default="data/train.csv")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1, help="concurrent API/CLI calls")
    parser.add_argument("--model", default=None, help="defaults per --provider if omitted")
    parser.add_argument("--out-dir", default="outputs/weak_labeling")
    parser.add_argument("--output", default="data/processed/weak_labels.csv", help="full mode only")
    parser.add_argument("--limit", type=int, default=None, help="full mode only - cap rows processed, for smoke testing")
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    if args.mode == "validate":
        validate(args)
    else:
        full_run(args)
