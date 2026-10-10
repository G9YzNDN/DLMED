"""Regenerate Topic 2 tables from committed results; Python standard library only.

This aggregates recorded experiment metrics. It does not run a trained model.
Run from any directory: python /path/to/DLMED/classification/report.py
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

SOURCE_COMMIT = "74170a0e5036fc1af0bae10b25a5196249a0add0"
EXPERIMENTS = ("A_cls", "C_mtl_equal", "D_mtl_uncert")
LABELS = {"A_cls": "A: Classification only", "C_mtl_equal": "C: Multi-task equal",
          "D_mtl_uncert": "D: Multi-task uncertainty"}
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1",
           "f1_meningioma", "f1_glioma", "f1_pituitary")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_runs(rows):
    runs = {exp: {} for exp in EXPERIMENTS}
    for row in rows:
        exp = row.get("experiment")
        if exp not in runs:
            continue
        seed = int(row["seed"])
        if seed in runs[exp]:
            raise ValueError(f"Duplicate experiment/seed: {exp}/{seed}")
        values = {metric: float(row[metric]) for metric in METRICS}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError(f"Non-finite or out-of-range metric: {exp}/{seed}")
        macro = statistics.mean(values[f"f1_{c}"] for c in ("meningioma", "glioma", "pituitary"))
        if not math.isclose(macro, values["macro_f1"], abs_tol=1e-8):
            raise ValueError(f"Macro-F1 differs from average per-class F1: {exp}/{seed}")
        runs[exp][seed] = values
    seeds = set(runs["A_cls"])
    if len(seeds) < 2 or any(set(runs[exp]) != seeds for exp in EXPERIMENTS):
        raise ValueError("A/C/D need the same seed set with at least two runs each")
    return runs, sorted(seeds)


def aggregate(runs, seeds):
    summary = []
    for exp in EXPERIMENTS:
        row = {"experiment": exp, "n_seeds": len(seeds)}
        for metric in METRICS:
            vals = [runs[exp][seed][metric] for seed in seeds]
            row[f"{metric}_mean"] = statistics.mean(vals)
            row[f"{metric}_std"] = statistics.stdev(vals)
        summary.append(row)
    paired = []
    for exp in EXPERIMENTS[1:]:
        for seed in seeds:
            paired.append({"comparison": f"{exp} - A_cls", "seed": seed,
                           "metric": "macro_f1",
                           "difference": runs[exp][seed]["macro_f1"] - runs["A_cls"][seed]["macro_f1"]})
    return summary, paired


def check_reference(summary, reference):
    indexed = {}
    for row in reference:
        exp = row["experiment"]
        if exp in indexed:
            raise ValueError(f"Duplicate experiment in reference: {exp}")
        indexed[exp] = row
    for row in summary:
        exp = row["experiment"]
        if exp not in indexed or int(indexed[exp]["n_seeds"]) != row["n_seeds"]:
            raise ValueError(f"Missing experiment or seed-count mismatch in reference: {exp}")
        for metric in METRICS:
            expected = f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"
            if indexed[exp][metric].strip() != expected:
                raise ValueError(f"Reference mismatch for {exp}/{metric}: expected {expected}")


def patient_errors(rows, runs, seeds):
    totals, patient_rows = {}, {}
    seen = set()
    for row in rows:
        exp = row["experiment"]
        if exp not in EXPERIMENTS:
            continue
        key = (exp, row["patient"])
        if key in seen:
            raise ValueError(f"Duplicate patient error row: {key}")
        seen.add(key)
        count, support = int(row["errors"]), int(row["slices_x_seeds"])
        if count < 0 or count > support:
            raise ValueError(f"Invalid patient error count: {key}")
        totals[exp] = totals.get(exp, 0) + count
        patient_rows.setdefault(exp, []).append(row)
    for exp in EXPERIMENTS:
        if exp not in totals:
            raise ValueError(f"Missing patient error rows: {exp}")
    error_rate_sum = sum(1 - runs["A_cls"][s]["accuracy"] for s in seeds)
    if error_rate_sum <= 0:
        raise ValueError("Cannot check error counts against a zero recorded error rate")
    inferred_size = totals["A_cls"] / error_rate_sum
    n_slices = round(inferred_size)
    if not math.isclose(inferred_size, n_slices, abs_tol=1e-6):
        raise ValueError("Patient error totals do not imply an integer test-set size")
    for exp in EXPERIMENTS:
        expected = sum(1 - runs[exp][s]["accuracy"] for s in seeds) * n_slices
        if not math.isclose(expected, totals[exp], abs_tol=1e-6):
            raise ValueError(f"Patient errors disagree with recorded accuracy: {exp}")
    return totals, patient_rows, n_slices


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def report_text(summary, paired, totals, patient_rows, n_slices, source_commit):
    def fmt(row, metric):
        return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"

    n_seeds = summary[0]["n_seeds"]
    lines = ["# Classification results", "",
             "Generated by `report.py` from recorded results. No new model training or inference was performed.", "",
             f"Source experiment commit: `{source_commit}`. Metric unit: one MRI slice. "
             f"{n_seeds} seed runs on the same {n_slices} test slices; the committed split has 29 patient groups.", "",
             "## Overall classification", "",
             f"Mean ± sample standard deviation across {n_seeds} seeds. SD describes training variability on this fixed split, not a confidence interval.", "",
             "| Experiment | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| " + LABELS[row["experiment"]] + " | " + " | ".join(
            fmt(row, metric) for metric in METRICS[:4]) + " |")
    lines += ["", "## Per-class F1", "",
              "| Experiment | Meningioma | Glioma | Pituitary |", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append("| " + LABELS[row["experiment"]] + " | " + " | ".join(
            fmt(row, metric) for metric in METRICS[4:]) + " |")
    lines += ["", "## Matched-seed Macro-F1 differences", "",
              "| Comparison | Per-seed difference | Mean difference | Positive seeds |", "|---|---|---:|---:|"]
    for exp in EXPERIMENTS[1:]:
        rs = [r for r in paired if r["comparison"] == f"{exp} - A_cls"]
        diffs = ", ".join(f"seed {r['seed']}: {r['difference']:+.4f}" for r in rs)
        mean = statistics.mean(r["difference"] for r in rs)
        wins = sum(r["difference"] > 0 for r in rs)
        lines.append(f"| {exp} − A_cls | {diffs} | {mean:+.4f} | {wins}/{len(rs)} |")
    lines += ["", "## Error concentration", "",
              "Error counts pool repeated predictions across seeds. Each patient row in the source contains at least one error.", "",
              "| Experiment | All errors | Errors from patient group 103673 | Share of all errors |",
              "|---|---:|---:|---:|"]
    for exp in EXPERIMENTS:
        count = sum(int(r["errors"]) for r in patient_rows[exp] if r["patient"] == "103673")
        lines.append(f"| {LABELS[exp]} | {totals[exp]} | {count} | {count / totals[exp]:.1%} |")
    lines += ["", "## Interpretation for Topic 2", "",
              "- Equal-weight joint learning (C) has higher recorded Macro-F1 than A in every seed in this experiment set.",
              "- Meningioma has the lowest mean per-class F1 in A, C and D on this split.",
              "- Group 103673 contributes 18 meningioma slices; the patient-error source reports them as pituitary in every seed of A/C/D (54 repeated errors per experiment).",
              "- The original report suggests a location/appearance explanation. That is an interpretation, not a confirmed clinical cause or a demonstrated model mechanism.",
              "- Segmentation supervision may improve useful shared features; the existing results do not establish why classification improves.", "",
              "## Confusion matrix", "",
              "![Classification confusion matrices](../results/figures/3_confusion_matrices.png)", "",
              f"Rows = true classes; columns = predicted classes. Each panel pools {n_slices * n_seeds:,} predictions over {n_seeds} seeds on the same {n_slices} slices. These are not independent images/patients. Aggregate scores cannot reconstruct the underlying matrix; this is the existing project figure.", "",
              "## Limits and reproducibility", "",
              "- Only one fixed patient-group split, 29 test groups and three recorded seeds; no formal statistical significance test or equivalence test was performed.",
              "- The original runs preceded the Albumentations-seeding fix; seed labels did not control augmentation draws. Corrected seeding applies to new training only, and the recorded-seed differences above are descriptive comparisons.",
              "- The learning rate was chosen on C only and reused for A/D; checkpoint selection also differs between single-task and joint models.",
              "- Suffix-based patient grouping is a conservative assumption about identifiers, not independently verified patient identity.",
              "- Classification covers three tumor types and has no healthy class; per-slice softmax probabilities were not evaluated for calibration.",
              "- All difficult cases remain in the full-test-set comparison. Excluding a patient after inspecting errors is a sensitivity analysis, not the primary result.",
              "- The script checks arithmetic against `results/summary_test.csv` and error totals against accuracy. Training checkpoints, original logs and per-image predictions are not committed, so table agreement does not independently verify training.", "",
              "Sources: [all runs](../results/all_runs_test.csv), [patient errors](../results/errors_by_patient.csv), "
              "[reference summary](../results/summary_test.csv), [original report](../results/RESULTS.md).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT,
                        help="Commit containing the source experiment results (recorded provenance)")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    paths = {name: root / "results" / name for name in (
        "all_runs_test.csv", "errors_by_patient.csv", "summary_test.csv")}
    runs, seeds = validate_runs(read_csv(paths["all_runs_test.csv"]))
    summary, paired = aggregate(runs, seeds)
    check_reference(summary, read_csv(paths["summary_test.csv"]))
    totals, patient_rows, n_slices = patient_errors(read_csv(paths["errors_by_patient.csv"]), runs, seeds)
    # This study package is tied to the reported split and known patient analysis.
    if len(seeds) != 3 or n_slices != 436:
        raise ValueError("This study package expects 3 seeds / 436 test slices; update the written notes for new experiments")
    for exp in EXPERIMENTS:
        target = [r for r in patient_rows[exp] if r["patient"] == "103673"]
        if len(target) != 1 or target[0]["class_name"] != "meningioma" or int(target[0]["errors"]) != 54 or int(target[0]["slices_x_seeds"]) != 54 or target[0]["predicted_as"] != "pituitary (54)":
            raise ValueError(f"Known-patient analysis changed for {exp}; update the written notes")
    provenance = {
        "source_experiment_commit": args.source_commit,
        "sources": {f"results/{name}": hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in paths.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "arithmetic aggregation of committed per-run metrics; sample SD (ddof=1)",
        "seed_labels": seeds, "test_slices": n_slices,
        "checks": ["finite metrics in [0,1]", "unique matching seeds",
                   "Macro-F1 equals mean per-class F1", "means and SD match reference at 4 decimals",
                   "patient error totals agree with accuracy", "known-patient analysis unchanged"],
        "new_training_performed": False, "new_inference_performed": False,
    }
    outputs = {
        "RESULTS.md": report_text(summary, paired, totals, patient_rows, n_slices, args.source_commit),
        "summary.csv": csv_text(summary), "paired.csv": csv_text(paired),
        "provenance.json": json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
    }
    out = args.out_dir or root / "classification"
    out.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (out / name).write_text(content, encoding="utf-8")
    print(f"Validated A/C/D ({len(seeds)} seeds each); wrote {len(outputs)} reports to {out}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, csv.Error) as exc:
        print(f"Classification report error: {exc}", file=sys.stderr)
        raise SystemExit(1)
