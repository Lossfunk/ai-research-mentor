#!/usr/bin/env python3
"""Compute robustness checks without new model calls.

Outputs are written to icml-evaluation-results/robustness_checks/.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "robustness_checks"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_md(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        f.write(text)


def pearson(x, y):
    if len(x) != len(y) or len(x) == 0:
        return None
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def rankdata(values):
    # Average ranks for ties (1-based ranks)
    sorted_vals = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j][0] == sorted_vals[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_vals[k][1]] = avg_rank
        i = j
    return ranks


def spearman(x, y):
    if len(x) != len(y) or len(x) == 0:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson(rx, ry)


def auc(scores, labels):
    # labels: 1 for positive, 0 for negative
    pairs = list(zip(scores, labels))
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return None
    # Mann–Whitney U based AUC
    ranks = rankdata([s for s, _ in pairs])
    rank_sum_pos = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def binom_two_sided_p(k, n):
    # exact two-sided binomial p-value with p=0.5
    if n == 0:
        return None
    # compute tail probability for k or more extreme
    p = 0.0
    prob = lambda x: math.comb(n, x) * (0.5 ** n)
    # two-sided: sum probs with prob <= prob(k)
    pk = prob(k)
    for x in range(0, n + 1):
        if prob(x) <= pk + 1e-12:
            p += prob(x)
    return min(1.0, p)


def fisher_exact_two_sided(a, b, c, d):
    # 2x2 table: [[a,b],[c,d]]
    # compute two-sided Fisher exact p-value
    def hypergeom_p(a_, b_, c_, d_):
        return (math.comb(a_ + b_, a_) * math.comb(c_ + d_, c_)) / math.comb(a_ + b_ + c_ + d_, a_ + c_)

    observed = hypergeom_p(a, b, c, d)
    p = 0.0
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    for a_ in range(max(0, col1 - row2), min(col1, row1) + 1):
        b_ = row1 - a_
        c_ = col1 - a_
        d_ = row2 - c_
        p_ = hypergeom_p(a_, b_, c_, d_)
        if p_ <= observed + 1e-12:
            p += p_
    return min(1.0, p)


def bootstrap_ci(diffs, iters=10000, seed=7):
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(iters):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(s))
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[int(0.975 * iters)]
    return [lo, hi]


def load_human_votes():
    votes = []
    for f in (ROOT / "human-baseline-votes").glob("*.csv"):
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if not r.get("winner"):
                    continue
                votes.append(r)
    return votes


def load_llm_scores():
    base = ROOT / "analysis_reports"
    system_dirs = {
        "mentor": base / "mentor",
        "gpt5": base / "gpt-5-baseline",
        "claude": base / "sonnet-4.5-baseline",
        "gemini": base / "gemini-baseline",
    }
    scores = {k: {} for k in system_dirs}
    stamps = {k: {} for k in system_dirs}
    for system, dir_path in system_dirs.items():
        for path in dir_path.rglob("*_judges.json"):
            data = read_json(path)
            prompt_id = data.get("prompt_id")
            metrics = data.get("metrics", {})
            hol = metrics.get("holistic_score", {})
            score = hol.get("score")
            stamp = data.get("generated_at") or data.get("run_timestamp") or ""
            if not prompt_id or score is None:
                continue
            prev_stamp = stamps[system].get(prompt_id, "")
            if stamp >= prev_stamp:
                scores[system][prompt_id] = score
                stamps[system][prompt_id] = stamp
    return scores


def load_raw_lengths():
    raw_root = ROOT / "raw_logs"
    system_dirs = {
        "mentor": raw_root / "mentor",
        "gpt5": raw_root / "gpt-5-baseline",
        "claude": raw_root / "sonnet-4.5-baseline",
        "gemini": raw_root / "gemini-baseline",
    }
    lengths = {k: {} for k in system_dirs}
    for system, dir_path in system_dirs.items():
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*.txt"):
            # Extract prompt_id like stage_a_01
            prompt_id = None
            stem = path.stem
            # scan for pattern stage_[a-f]_## (length 10)
            for i in range(len(stem) - 9):
                sub = stem[i : i + 10]
                if (
                    sub.startswith("stage_")
                    and sub[6].isalpha()
                    and sub[7] == "_"
                    and sub[8].isdigit()
                    and sub[9].isdigit()
                ):
                    prompt_id = sub
                    break
            if not prompt_id:
                # try simple split with first three parts
                parts = stem.split("_")
                if len(parts) >= 3 and parts[0] == "stage":
                    prompt_id = "_".join(parts[:3])
            if prompt_id is None:
                continue
            text = path.read_text()
            words = [w for w in text.split() if w.strip()]
            lengths[system][prompt_id] = {
                "chars": len(text),
                "words": len(words),
            }
    return lengths


def compute_human_llm_alignment(votes, scores):
    # only mentor vs baseline comparisons
    rows = [v for v in votes if v["pair_type"].startswith("mentor_vs_")]
    by_matchup = {}
    all_records = []
    for v in rows:
        prompt_id = v["prompt_id"]
        baseline = v["pair_type"].replace("mentor_vs_", "")
        # map baseline names
        baseline_map = {"gpt5": "gpt5", "claude": "claude", "gemini": "gemini"}
        baseline = baseline_map.get(baseline, baseline)
        if prompt_id not in scores["mentor"] or prompt_id not in scores.get(baseline, {}):
            continue
        m = scores["mentor"][prompt_id]
        b = scores[baseline][prompt_id]
        diff = m - b
        human_winner = v["winner"].strip()
        if human_winner not in {"mentor", baseline, "tie"}:
            continue
        all_records.append({
            "prompt_id": prompt_id,
            "baseline": baseline,
            "diff": diff,
            "human_winner": human_winner,
        })
        by_matchup.setdefault(baseline, []).append(all_records[-1])

    def summarize(records):
        # accuracy on decisive cases
        decisive = [r for r in records if r["human_winner"] != "tie"]
        correct = 0
        for r in decisive:
            pred = "tie"
            if r["diff"] > 0:
                pred = "mentor"
            elif r["diff"] < 0:
                pred = r["baseline"]
            if pred == r["human_winner"]:
                correct += 1
        acc = correct / len(decisive) if decisive else None
        labels = [1 if r["human_winner"] == "mentor" else 0 for r in decisive]
        diffs = [r["diff"] for r in decisive]
        return {
            "n_total": len(records),
            "n_decisive": len(decisive),
            "accuracy": acc,
            "pearson_r": pearson(diffs, labels) if decisive else None,
            "spearman_r": spearman(diffs, labels) if decisive else None,
            "auc": auc(diffs, labels) if decisive else None,
        }

    summary = {
        "overall": summarize(all_records),
        "by_baseline": {k: summarize(v) for k, v in by_matchup.items()},
    }
    return summary


def compute_order_bias(votes):
    rows = [v for v in votes if v["pair_type"].startswith("mentor_vs_")]
    # mentor is always system_a in this dataset; displayed_position indicates where mentor was shown.
    def win_rate(records):
        dec = [r for r in records if r["winner"] != "tie"]
        wins = sum(1 for r in dec if r["winner"] == "mentor")
        return wins, len(dec), wins / len(dec) if dec else None

    by_pos = {
        "A": [r for r in rows if r["displayed_position"] == "A"],
        "B": [r for r in rows if r["displayed_position"] == "B"],
    }
    wins_a, n_a, rate_a = win_rate(by_pos["A"])
    wins_b, n_b, rate_b = win_rate(by_pos["B"])

    # 2x2 table for Fisher exact
    # rows: mentor shown A vs B; cols: mentor win vs baseline win
    a = wins_a
    b = n_a - wins_a
    c = wins_b
    d = n_b - wins_b
    p = fisher_exact_two_sided(a, b, c, d) if n_a and n_b else None

    return {
        "displayed_A": {"wins": wins_a, "decisive": n_a, "win_rate": rate_a},
        "displayed_B": {"wins": wins_b, "decisive": n_b, "win_rate": rate_b},
        "fisher_two_sided_p": p,
    }


def compute_length_bias(scores, lengths):
    summary = {}
    for system in scores:
        xs = []  # lengths
        ys = []  # holistic scores
        xw = []
        for prompt_id, score in scores[system].items():
            if prompt_id not in lengths.get(system, {}):
                continue
            xs.append(lengths[system][prompt_id]["chars"])
            xw.append(lengths[system][prompt_id]["words"])
            ys.append(score)
        summary[system] = {
            "n": len(ys),
            "pearson_r_chars": pearson(xs, ys) if ys else None,
            "spearman_r_chars": spearman(xs, ys) if ys else None,
            "pearson_r_words": pearson(xw, ys) if ys else None,
            "spearman_r_words": spearman(xw, ys) if ys else None,
        }
    return summary


def compute_multi_turn_significance():
    # sign test across paired scenarios
    path = ROOT / "holistic_scoring_v2" / "holistic_results.csv"
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Map system_id -> scenario_id -> holistic_score
    sys_map = {
        "mentor": "openrouter:moonshotai/kimi-k2-thinking",
        "gpt5": "openrouter:openai/gpt-5",
        "claude": "openrouter:anthropic/claude-sonnet-4.5",
        "gemini": "openrouter:google/gemini-3-pro-preview",
    }
    per_sys = {k: {} for k in sys_map}
    for r in rows:
        for key, sys_id in sys_map.items():
            if r["system_id"] == sys_id:
                per_sys[key][r["scenario_id"]] = float(r["holistic_score"])

    mentor = per_sys["mentor"]
    results = {}
    for base in ["gpt5", "claude", "gemini"]:
        diffs = []
        wins = losses = ties = 0
        for scenario, mscore in mentor.items():
            if scenario not in per_sys[base]:
                continue
            bscore = per_sys[base][scenario]
            diff = mscore - bscore
            diffs.append(diff)
            if diff > 0:
                wins += 1
            elif diff < 0:
                losses += 1
            else:
                ties += 1
        n_decisive = wins + losses
        p = binom_two_sided_p(wins, n_decisive) if n_decisive else None
        results[base] = {
            "n": len(diffs),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_diff": mean(diffs) if diffs else None,
            "median_diff": median(diffs) if diffs else None,
            "sign_test_p": p,
            "bootstrap_ci_mean_diff": bootstrap_ci(diffs) if diffs else None,
        }
    return results


def summarize_user_study():
    path = ROOT / "METIS human eval results.csv"
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    # Identify Likert columns (numeric)
    likert_cols = [
        "The research mentor was easy to use. ",
        "The advice given by the mentor was helpful and relevant.",
        "The mentor understood my research goals.",
        "Overall, using this tool was a positive experience.",
    ]
    reuse_col = "Would you use a tool like this for your future research projects?"
    hours_col = "Approximately how many hours did you spend on this task from start to finish?"

    likert = {}
    for col in likert_cols:
        vals = [float(r[col]) for r in rows if r.get(col)]
        likert[col] = {
            "n": len(vals),
            "mean": mean(vals) if vals else None,
            "median": median(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    reuse_counts = {}
    for r in rows:
        ans = r.get(reuse_col, "").strip()
        if ans:
            reuse_counts[ans] = reuse_counts.get(ans, 0) + 1

    hours = [float(r[hours_col]) for r in rows if r.get(hours_col)]
    hours_stats = {
        "n": len(hours),
        "mean": mean(hours) if hours else None,
        "median": median(hours) if hours else None,
        "min": min(hours) if hours else None,
        "max": max(hours) if hours else None,
    }

    return {
        "n_participants": len(rows),
        "likert": likert,
        "reuse_counts": reuse_counts,
        "hours": hours_stats,
    }


def main():
    ensure_dir(OUT_ROOT)

    votes = load_human_votes()
    scores = load_llm_scores()
    lengths = load_raw_lengths()

    # Human-LLM alignment
    alignment = compute_human_llm_alignment(votes, scores)
    write_json(OUT_ROOT / "human_llm_alignment" / "alignment_summary.json", alignment)

    # Order bias
    order_bias = compute_order_bias(votes)
    write_json(OUT_ROOT / "order_bias" / "order_bias_summary.json", order_bias)

    # Length bias
    length_bias = compute_length_bias(scores, lengths)
    write_json(OUT_ROOT / "length_bias" / "length_bias_summary.json", length_bias)

    # Multi-turn paired significance
    multi_turn = compute_multi_turn_significance()
    write_json(OUT_ROOT / "multi_turn_significance" / "sign_test_summary.json", multi_turn)

    # User study summary
    user_study = summarize_user_study()
    write_json(OUT_ROOT / "user_study_summary" / "metis_user_study_summary.json", user_study)

    # Write lightweight MD summaries
    write_md(
        OUT_ROOT / "human_llm_alignment" / "alignment_summary.md",
        f"""# Human ↔ LLM Alignment (Single-turn)

Overall accuracy (decisive only): {alignment['overall']['accuracy']:.3f}
Pearson r: {alignment['overall']['pearson_r']:.3f}
Spearman r: {alignment['overall']['spearman_r']:.3f}
AUC: {alignment['overall']['auc']:.3f}

By baseline:
{json.dumps(alignment['by_baseline'], indent=2)}
""",
    )

    write_md(
        OUT_ROOT / "order_bias" / "order_bias_summary.md",
        f"""# Human Order Bias (Mentor display A vs B)

Displayed A win rate: {order_bias['displayed_A']['win_rate']:.3f} (n={order_bias['displayed_A']['decisive']})
Displayed B win rate: {order_bias['displayed_B']['win_rate']:.3f} (n={order_bias['displayed_B']['decisive']})
Fisher two-sided p: {order_bias['fisher_two_sided_p']}
""",
    )

    write_md(
        OUT_ROOT / "length_bias" / "length_bias_summary.md",
        "# Length Bias (Holistic score vs response length)\n\n"
        + json.dumps(length_bias, indent=2)
        + "\n",
    )

    write_md(
        OUT_ROOT / "multi_turn_significance" / "sign_test_summary.md",
        "# Multi-turn Paired Sign Tests (Mentor vs baseline)\n\n"
        + json.dumps(multi_turn, indent=2)
        + "\n",
    )

    write_md(
        OUT_ROOT / "user_study_summary" / "metis_user_study_summary.md",
        "# METIS User Study Summary\n\n"
        + json.dumps(user_study, indent=2)
        + "\n",
    )


if __name__ == "__main__":
    main()
