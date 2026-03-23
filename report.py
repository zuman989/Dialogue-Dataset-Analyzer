"""
report.py — Generate HTML report and JSON export.
Usage: python report.py
"""

import os, json, base64, io
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocess import load_data, preprocess_dialogs
from src.feature_extraction import (
    get_speaker_stats, compare_speakers,
    collect_all_speaker_stats, compute_population_stats,
    FEATURES, FEATURE_DISPLAY_NAMES,
)
from src.persona import generate_persona

# ── Config ──

DATA_PATH = "data/train.csv"
NUM_CONVERSATIONS = 10
OUTPUT_DIR = "reports"

# Chart palette (warm orange theme)
C_A = "#e8590c"     # Speaker A — burnt orange
C_B = "#f59f00"     # Speaker B — amber
C_POS = "#2b8a3e"   # positive
C_NEG = "#e03131"   # negative
C_NEU = "#868e96"   # neutral
PALETTE = ["#e8590c", "#f59f00", "#1c7ed6", "#2b8a3e", "#ae3ec9",
           "#e03131", "#0ca678", "#f08c00", "#845ef7"]


def _to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _clean_ax(ax, title=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ── Chart generators ──

def chart_traits(personas):
    counts = Counter()
    for p in personas:
        for spk in ["A", "B"]:
            for t in p[spk]["traits"]:
                counts[t] += 1
    if not counts:
        return ""
    labels, vals = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels[::-1], vals[::-1], color=PALETTE[:len(labels)], height=0.55)
    _clean_ax(ax, "Personality Trait Frequency")
    fig.tight_layout()
    return _to_b64(fig)


def chart_sentiment(stats_list):
    n = len(stats_list)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    w = 0.2
    ax.bar(x - 1.5*w, [s["A"]["positive"] for s in stats_list], w, label="A positive", color=C_A)
    ax.bar(x - 0.5*w, [s["B"]["positive"] for s in stats_list], w, label="B positive", color=C_B)
    ax.bar(x + 0.5*w, [s["A"]["negative"] for s in stats_list], w, label="A negative", color=C_NEG, alpha=0.7)
    ax.bar(x + 1.5*w, [s["B"]["negative"] for s in stats_list], w, label="B negative", color=C_NEG, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i+1}" for i in range(n)])
    ax.legend(fontsize=8, ncol=4, frameon=False)
    _clean_ax(ax, "Sentiment per Conversation")
    fig.tight_layout()
    return _to_b64(fig)


def chart_line(stats_list, key, title):
    n = len(stats_list)
    x = list(range(1, n + 1))
    va = [s["A"][key] for s in stats_list]
    vb = [s["B"][key] for s in stats_list]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, va, "o-", color=C_A, lw=2, ms=5, label="Speaker A")
    ax.plot(x, vb, "s-", color=C_B, lw=2, ms=5, label="Speaker B")
    ax.fill_between(x, va, alpha=0.08, color=C_A)
    ax.fill_between(x, vb, alpha=0.08, color=C_B)
    ax.set_xticks(x)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _clean_ax(ax, title)
    fig.tight_layout()
    return _to_b64(fig)


def chart_scatter(stats_list):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter([s["A"]["formality"] for s in stats_list],
               [s["A"]["avg_tree_depth"] for s in stats_list],
               c=C_A, s=70, alpha=0.8, label="Speaker A", edgecolors="white", lw=1)
    ax.scatter([s["B"]["formality"] for s in stats_list],
               [s["B"]["avg_tree_depth"] for s in stats_list],
               c=C_B, s=70, alpha=0.8, label="Speaker B", edgecolors="white", lw=1)
    ax.set_xlabel("Formality (0–100)", fontsize=10)
    ax.set_ylabel("Sentence Complexity", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.15)
    _clean_ax(ax, "Formality vs Complexity")
    fig.tight_layout()
    return _to_b64(fig)


# ── HTML helpers ──

def mentions_html(s):
    if not s["entities"]:
        return '<span class="text-muted">—</span>'
    seen, parts = set(), []
    for text, _, readable in s["entities"]:
        if text not in seen:
            seen.add(text)
            parts.append(f'<span class="badge bg-light text-dark border me-1">{text} ({readable})</span>')
    return "".join(parts[:4])


def trait_html(t):
    colors = {
        "curious": "#7048e8", "expressive": "#e8590c", "enthusiastic": "#f59f00",
        "critical": "#e03131", "withdrawn": "#868e96", "articulate": "#0ca678",
        "reserved": "#495057", "assertive": "#d6336c", "analytical": "#1c7ed6",
    }
    c = colors.get(t, "#495057")
    return f'<span class="badge rounded-pill" style="background:{c};font-size:0.78rem">{t}</span>'


def convo_html(idx, convo, stats, comp, persona):
    a, b = stats["A"], stats["B"]

    # dialogue
    msgs = ""
    for m in convo:
        color = C_A if m["speaker"] == "A" else C_B
        msgs += f'<div class="mb-1 ps-3" style="border-left:3px solid {color}"><small><b style="color:{color}">{m["speaker"]}</b></small> {m["text"]}</div>\n'

    # persona cards
    def pcard(spk):
        p = persona[spk]
        color = C_A if spk == "A" else C_B
        traits = " ".join(trait_html(t) for t in p["traits"])
        topics = ""
        if p["topics"]:
            types_str = ", ".join(f'{c} {l.lower()}' for l, c in p["topics"]["types"].items())
            topics = f'<tr><td class="text-muted">Talks about</td><td>{types_str}</td></tr>'
        return f"""
        <div class="col-md-6">
        <div class="card h-100" style="border-top:3px solid {color}">
        <div class="card-body">
            <h6><span class="badge rounded-circle" style="background:{color}">{spk}</span> Speaker {spk}</h6>
            <table class="table table-sm table-borderless mb-2" style="font-size:0.85rem">
            <tr><td class="text-muted">Communication</td><td>{p['style']}</td></tr>
            <tr><td class="text-muted">Engagement</td><td>{p['engagement']}</td></tr>
            <tr><td class="text-muted">Emotional tone</td><td>{p['tone']}</td></tr>
            <tr><td class="text-muted">Speech style</td><td>{p['speech_style']}</td></tr>
            <tr><td class="text-muted">Role</td><td>{p['role']}</td></tr>
            {topics}
            </table>
            <div>{traits}</div>
        </div></div></div>"""

    # stat row helper
    def sr(label, va, vb, fmt="{}"):
        return f'<tr><td>{label}</td><td class="text-end font-monospace">{fmt.format(va)}</td><td class="text-end font-monospace">{fmt.format(vb)}</td></tr>'

    at = f"{comp['A_dominance']:.0%}"
    bt = f"{comp['B_dominance']:.0%}"

    return f"""
    <div class="card mb-4" id="convo-{idx}">
    <div class="card-header bg-white"><h5 class="mb-0">Conversation {idx}</h5></div>
    <div class="card-body">

        <div class="border rounded p-3 mb-3" style="max-height:300px;overflow-y:auto;background:#fefcfb">{msgs}</div>

        <div class="table-responsive">
        <table class="table table-sm align-middle" style="font-size:0.87rem">
        <thead class="table-light">
            <tr><th></th><th class="text-end" style="color:{C_A}">Speaker A</th><th class="text-end" style="color:{C_B}">Speaker B</th></tr>
        </thead>
        <tbody>
            {sr("Messages", a['messages'], b['messages'])}
            {sr("Words", a['words'], b['words'])}
            {sr("Avg words/message", a['avg_length'], b['avg_length'], "{:.1f}")}
            {sr("Questions asked", a['questions'], b['questions'])}
            {sr("Vocabulary diversity", a['vocab_richness'], b['vocab_richness'], "{:.2f}")}
            <tr class="table-light"><td colspan="3"><small class="text-muted fw-bold">SENTIMENT</small></td></tr>
            {sr("Positive", a['positive'], b['positive'])}
            {sr("Negative", a['negative'], b['negative'])}
            {sr("Neutral", a['neutral'], b['neutral'])}
            {sr("Avg score", a['avg_sentiment'], b['avg_sentiment'], "{:+.2f}")}
            <tr class="table-light"><td colspan="3"><small class="text-muted fw-bold">STYLE</small></td></tr>
            {sr("Sentence complexity", a['avg_tree_depth'], b['avg_tree_depth'], "{:.1f}")}
            {sr("Formality", a['formality'], b['formality'], "{:.0f}/100")}
            {sr("Hedge words", a['hedges'], b['hedges'])}
            {sr("Certainty words", a['certainty_markers'], b['certainty_markers'])}
            <tr class="table-light"><td colspan="3"><small class="text-muted fw-bold">CONTENT</small></td></tr>
            <tr><td>Mentions</td><td>{mentions_html(a)}</td><td>{mentions_html(b)}</td></tr>
            <tr class="table-light"><td colspan="3"></td></tr>
            <tr><td>Turn balance</td><td class="text-end font-monospace">{at}</td><td class="text-end font-monospace">{bt}</td></tr>
        </tbody>
        </table>
        </div>

        <div class="row g-3 mt-1">
            {pcard("A")}
            {pcard("B")}
        </div>

    </div></div>"""


def build_html(population, analyses, charts, total):
    nav = "".join(f'<a href="#convo-{a["index"]}" class="btn btn-sm btn-outline-secondary">{a["index"]}</a>' for a in analyses)
    pop_rows = "".join(f'<tr><td>{FEATURE_DISPLAY_NAMES.get(f,f)}</td><td class="text-end font-monospace">{population[f]["mean"]:.2f}</td><td class="text-end font-monospace">{population[f]["std"]:.2f}</td></tr>' for f in FEATURES)
    convos = "".join(convo_html(a["index"], a["convo"], a["stats"], a["comp"], a["persona"]) for a in analyses)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dialogue Analysis Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
body {{ font-family: 'Inter', sans-serif; background: #f4f0eb; }}
.hero {{ background: linear-gradient(135deg, #e8590c, #f76707, #f59f00); color: white; padding: 48px 0; }}
.hero h1 {{ font-size: 2rem; font-weight: 700; }}
.stat-card {{ background: rgba(255,255,255,0.15); border-radius: 10px; padding: 18px; text-align: center; backdrop-filter: blur(8px); }}
.stat-card .num {{ font-size: 1.6rem; font-weight: 700; }}
.stat-card .label {{ font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }}
.nav-pills .btn {{ margin: 2px; }}
.chart-img {{ width: 100%; border-radius: 8px; }}
.card {{ border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-radius: 10px; }}
.card-header {{ border-bottom: 1px solid #eee; font-weight: 600; }}
.table td, .table th {{ vertical-align: middle; }}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
<div class="container">
    <h1>Dialogue Dataset Analysis</h1>
    <p class="mb-4 opacity-75">Persona extraction from DailyDialog — {total:,} conversations analyzed</p>
    <div class="row g-3">
        <div class="col-6 col-md-3"><div class="stat-card"><span class="num">{total:,}</span><br><span class="label">Conversations</span></div></div>
        <div class="col-6 col-md-3"><div class="stat-card"><span class="num">{len(analyses)}</span><br><span class="label">Detailed Analysis</span></div></div>
        <div class="col-6 col-md-3"><div class="stat-card"><span class="num">{population['messages']['mean']:.1f}</span><br><span class="label">Avg Msgs / Speaker</span></div></div>
        <div class="col-6 col-md-3"><div class="stat-card"><span class="num">{population['avg_sentiment']['mean']:+.2f}</span><br><span class="label">Avg Sentiment</span></div></div>
    </div>
</div>
</div>

<!-- Nav -->
<div class="sticky-top bg-white border-bottom py-2">
<div class="container d-flex gap-2 align-items-center flex-wrap">
    <span class="text-muted small fw-bold me-2">JUMP TO</span>
    <a href="#baseline" class="btn btn-sm btn-outline-secondary">Baseline</a>
    <a href="#charts" class="btn btn-sm btn-outline-secondary">Charts</a>
    {nav}
</div>
</div>

<div class="container py-4">

<!-- Baseline -->
<div class="card mb-4" id="baseline">
<div class="card-header bg-white">Dataset Baseline</div>
<div class="card-body">
    <p class="text-muted small">Average values across all {total:,} conversations</p>
    <div class="table-responsive">
    <table class="table table-sm table-hover" style="font-size:0.87rem">
    <thead class="table-light"><tr><th>Feature</th><th class="text-end">Average</th><th class="text-end">Spread</th></tr></thead>
    <tbody>{pop_rows}</tbody>
    </table>
    </div>
</div></div>

<!-- Charts -->
<div id="charts" class="mb-4">
<h5 class="fw-bold mb-3">Visual Overview</h5>
<div class="row g-3">
    <div class="col-md-7"><div class="card"><div class="card-body"><img class="chart-img" src="data:image/png;base64,{charts['traits']}"></div></div></div>
    <div class="col-md-5"><div class="card"><div class="card-body"><img class="chart-img" src="data:image/png;base64,{charts['scatter']}"></div></div></div>
    <div class="col-12"><div class="card"><div class="card-body"><img class="chart-img" src="data:image/png;base64,{charts['sentiment']}"></div></div></div>
    <div class="col-md-6"><div class="card"><div class="card-body"><img class="chart-img" src="data:image/png;base64,{charts['vocab']}"></div></div></div>
    <div class="col-md-6"><div class="card"><div class="card-body"><img class="chart-img" src="data:image/png;base64,{charts['complexity']}"></div></div></div>
</div></div>

<!-- Conversations -->
<h5 class="fw-bold mb-3">Conversation Analysis</h5>
{convos}

</div>

<footer class="text-center text-muted py-4 small border-top">
    Dialogue Dataset Analyzer &mdash; Python, spaCy, NLTK, VADER, Matplotlib
</footer>

</body></html>"""


# ── JSON export ──

def build_json(population, analyses):
    def clean(s):
        out = {}
        for k, v in s.items():
            if k == "unique_words": out[k] = list(v)
            elif k in ("entity_types","entity_types_readable","gt_acts","gt_emotions"): out[k] = dict(v)
            elif k == "entities": out[k] = [(t, r, rd) for t, r, rd in v]
            else: out[k] = v
        return out

    return {
        "population": population,
        "conversations": [{
            "index": a["index"],
            "stats": {"A": clean(a["stats"]["A"]), "B": clean(a["stats"]["B"])},
            "comparison": a["comp"],
            "persona": {spk: {k:v for k,v in a["persona"][spk].items() if not k.startswith("_")} for spk in ["A","B"]},
        } for a in analyses]
    }


# ── Main ──

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    processed = preprocess_dialogs(df)
    total = len(processed)

    print(f"Computing population stats ({total} conversations)...")
    all_stats, _ = collect_all_speaker_stats(processed)
    population = compute_population_stats(all_stats)

    print(f"Analyzing first {NUM_CONVERSATIONS} conversations...")
    analyses = []
    stats_list = []
    personas = []

    for i, convo in enumerate(processed[:NUM_CONVERSATIONS]):
        stats = get_speaker_stats(convo)
        comp = compare_speakers(stats)
        persona = generate_persona(stats, comp, population)
        analyses.append({"index": i+1, "convo": convo, "stats": stats, "comp": comp, "persona": persona})
        stats_list.append(stats)
        personas.append(persona)

    print("Generating charts...")
    charts = {
        "traits": chart_traits(personas),
        "sentiment": chart_sentiment(stats_list),
        "vocab": chart_line(stats_list, "vocab_richness", "Vocabulary Diversity"),
        "complexity": chart_line(stats_list, "avg_tree_depth", "Sentence Complexity"),
        "scatter": chart_scatter(stats_list),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Writing HTML...")
    with open(os.path.join(OUTPUT_DIR, "report.html"), "w", encoding="utf-8") as f:
        f.write(build_html(population, analyses, charts, total))

    print("Writing JSON...")
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(build_json(population, analyses), f, indent=2, default=str)

    print(f"\nDone! → reports/report.html & reports/report.json")


if __name__ == "__main__":
    main()