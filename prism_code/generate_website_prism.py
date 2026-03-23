#!/usr/bin/env python3
"""Generate a self-contained interactive HTML website for HHMM-GMM analysis.

Reads aggregate_transitions.json and aggregate_bottom.json, produces a single
index.html with embedded data and Plotly.js visualizations.

Usage:
    python generate_website.py \
        --top_json aggregate_results/aggregate_transitions.json \
        --bottom_json aggregate_bottom_results/aggregate_bottom.json \
        --output index.html
"""

import argparse
import json
import os

# ── Category constants (must match analysis pipeline) ────────────────
TAG_SHORT = ["FA", "SR", "AC", "UV"]
CAT_COLORS = {
    "FA": "#C48B9F", "SR": "#8B9DAA", "AC": "#8BAA92", "UV": "#C9A84C",
}
CAT_NAMES = {
    "FA": "Final Answer", "SR": "Setup & Retrieval",
    "AC": "Analysis & Computation", "UV": "Uncertainty & Verification",
}


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def generate_html(top_data, bottom_data, fixed_model=None, fixed_dataset=None):
    """Generate the full HTML string.

    When *fixed_model* and *fixed_dataset* are given the page is rendered for
    a single model/dataset combination: the filter bar is hidden and the
    model/dataset is pre-selected.
    """

    top_json = json.dumps(top_data) if top_data else "{}"
    bottom_json = json.dumps(bottom_data) if bottom_data else "{}"
    cat_colors_json = json.dumps(CAT_COLORS)
    cat_names_json = json.dumps(CAT_NAMES)
    tags_json = json.dumps(TAG_SHORT)

    is_fixed = fixed_model is not None and fixed_dataset is not None
    is_all_page = is_fixed and fixed_model == "_all_"
    individual_tab_btn = ("" if is_all_page
                          else '<button class="nav-tab" onclick="switchTab(\'individual\')">Individual Examples</button>')
    page_title = ("All Models & Datasets — PRISM Analysis" if is_all_page
                  else f"{fixed_model} / {fixed_dataset} — PRISM Analysis"
                  if is_fixed else "HHMM-GMM Reasoning Trace Analysis")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', system-ui, sans-serif;
        background: #F5F3F0; color: #2C2C2E; -webkit-font-smoothing: antialiased; }}

/* ── Navigation ─────────────────────────────── */
.nav {{ background: rgba(255,255,255,0.85); backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 0 32px; display: flex; align-items: center; gap: 24px;
        border-bottom: 1px solid #E8E4E0; position: sticky; top: 0; z-index: 100; }}
.nav h1 {{ font-size: 18px; color: #2C2C2E; padding: 16px 0; white-space: nowrap; font-weight: 700; letter-spacing: -0.3px; }}
.nav-tabs {{ display: flex; gap: 4px; }}
.nav-tab {{ padding: 10px 20px; cursor: pointer; border-radius: 20px;
            background: transparent; color: #8E8E93; font-size: 14px; font-weight: 500;
            border: none; transition: all 0.25s ease; }}
.nav-tab:hover {{ color: #2C2C2E; background: rgba(0,0,0,0.04); }}
.nav-tab.active {{ color: #fff; background: #7C8B9A; }}

/* ── Layout ─────────────────────────────────── */
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}

.section {{ background: #fff; border-radius: 16px; padding: 24px; margin-bottom: 20px;
            border: 1px solid #E8E4E0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.section h2 {{ font-size: 18px; margin-bottom: 4px; color: #2C2C2E; font-weight: 600; letter-spacing: -0.2px; }}
.section h3 {{ font-size: 15px; margin: 16px 0 8px; color: #6B6B70; }}
.fn {{ font-size: 12px; color: #8E8E93; font-weight: 400; margin-bottom: 14px; line-height: 1.5; }}
.section-compact {{ padding: 18px; }}

.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
.grid-4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }}
.grid-2-1 {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }}

/* ── Stat cards ─────────────────────────────── */
.stat-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.stat-card {{ background: #F9F8F6; border-radius: 12px; padding: 14px 18px; flex: 1; min-width: 120px;
              border: 1px solid #ECEAE6; }}
.stat-card .label {{ font-size: 11px; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 500; }}
.stat-card .value {{ font-size: 22px; font-weight: 700; color: #2C2C2E; margin-top: 4px; }}
.stat-card .sub {{ font-size: 11px; color: #AEAEB2; margin-top: 2px; }}

/* ── Plotly overrides ───────────────────────── */
.js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}

/* ── Matrix selector ────────────────────────── */
.btn-group {{ display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }}
.btn {{ padding: 7px 16px; border-radius: 20px; border: 1px solid #D6D3CE; background: #fff;
        color: #6B6B70; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s ease; }}
.btn:hover {{ border-color: #7C8B9A; color: #2C2C2E; }}
.btn.active {{ background: #7C8B9A; border-color: #7C8B9A; color: #fff; }}

/* ── Filter bar ────────────────────────────── */
.filter-bar {{ background: rgba(255,255,255,0.88); backdrop-filter: saturate(180%) blur(16px);
              -webkit-backdrop-filter: saturate(180%) blur(16px);
              padding: 8px 32px; display: flex; align-items: center; gap: 16px;
              border-bottom: 1px solid #E8E4E0; position: sticky; top: 51px; z-index: 99; }}
.filter-bar label {{ font-size: 12px; color: #8E8E93; font-weight: 500; }}
.filter-bar select {{ padding: 5px 10px; border-radius: 8px; border: 1px solid #D6D3CE;
                      font-size: 13px; color: #2C2C2E; background: #fff; cursor: pointer;
                      font-family: inherit; outline: none; transition: border-color 0.2s; }}
.filter-bar select:hover {{ border-color: #7C8B9A; }}
.filter-bar select:focus {{ border-color: #7C8B9A; box-shadow: 0 0 0 2px rgba(124,139,154,0.15); }}
.filter-status {{ font-size: 12px; color: #8E8E93; margin-left: auto; }}
.agg-note {{ font-size: 11px; color: #AEAEB2; font-style: italic; margin-top: 4px; }}

/* ── Timeline ───────────────────────────────── */
.example-card {{ background: #F9F8F6; border-radius: 12px; padding: 20px; margin-bottom: 12px;
                 border: 1px solid #ECEAE6; cursor: pointer; transition: all 0.25s ease; }}
.example-card:hover {{ border-color: #7C8B9A; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.example-card.expanded {{ border-color: #7C8B9A; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
.example-header {{ display: flex; justify-content: space-between; align-items: center; }}
.example-header .tag {{ padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.tag-correct {{ background: rgba(91,141,184,0.15); color: #3A6B96; }}
.tag-long {{ background: rgba(212,136,92,0.15); color: #A06030; }}
.tag-short {{ background: rgba(196,177,118,0.15); color: #9A8B55; }}

.example-question {{ font-size: 14px; color: #6B6B70; margin: 8px 0; line-height: 1.5;
                     max-height: 60px; overflow: hidden; text-overflow: ellipsis; }}
.example-card.expanded .example-question {{ max-height: none; }}

.timeline {{ display: flex; margin: 12px 0; overflow-x: auto; gap: 2px; padding: 8px 4px 4px 4px; }}
.timeline-step {{ min-width: 28px; height: 34px; display: flex; align-items: center; justify-content: center;
                  font-size: 10px; font-weight: 700; color: #fff; border-radius: 6px;
                  cursor: pointer; position: relative; transition: transform 0.15s ease; flex-shrink: 0; }}
.timeline-step:hover {{ transform: scaleY(1.25); z-index: 10; }}

.tooltip {{ display: none; position: absolute; bottom: 110%; left: 50%; transform: translateX(-50%);
            background: #fff; border: 1px solid #E8E4E0; border-radius: 10px; padding: 10px 14px;
            white-space: nowrap; font-size: 12px; color: #2C2C2E; z-index: 1000;
            box-shadow: 0 4px 16px rgba(0,0,0,0.12); pointer-events: none; }}
.tooltip::after {{ content: ''; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
                   border: 6px solid transparent; border-top-color: #E8E4E0; }}
.timeline-step:hover .tooltip {{ display: block; }}

.layer-strip {{ display: flex; gap: 1px; overflow-x: auto; padding: 4px 0; }}
.layer-cell {{ width: 14px; height: 22px; display: flex; align-items: center; justify-content: center;
              font-size: 7px; color: rgba(255,255,255,0.9); border-radius: 3px; flex-shrink: 0;
              cursor: default; position: relative; }}
.layer-cell .layer-tip {{ display: none; position: absolute; bottom: 110%; left: 50%; transform: translateX(-50%);
                          background: #fff; border: 1px solid #E8E4E0; border-radius: 8px; padding: 6px 10px;
                          white-space: nowrap; font-size: 11px; color: #2C2C2E; z-index: 100;
                          box-shadow: 0 2px 10px rgba(0,0,0,0.1); pointer-events: none; }}
.layer-cell:hover .layer-tip {{ display: block; }}

/* ── Step trajectory panel ──────────────── */
.traj-chip {{ display:flex; align-items:center; gap:6px; padding:5px 8px; border-radius:8px;
              margin:2px 0; cursor:pointer; font-size:11px; border:1.5px solid transparent;
              transition: all 0.15s ease; user-select:none; }}
.traj-chip:hover {{ background:rgba(124,139,154,0.1); }}
.traj-chip.active {{ border-color:#7C8B9A; background:rgba(124,139,154,0.13); }}
.traj-chip-dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
.traj-group-label {{ font-size:10px; font-weight:700; letter-spacing:0.05em; color:#8E8E93;
                     margin:8px 0 4px; padding:0 2px; }}

/* ── Step detail panel (click-to-open) ───── */
.step-detail {{ display: none; margin: 10px 0; padding: 16px; background: #fff; border: 1px solid #D6D3CE;
                border-radius: 12px; animation: slideDown 0.25s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.step-detail.open {{ display: block; }}
@keyframes slideDown {{ from {{ opacity: 0; transform: translateY(-8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
.step-detail-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
.step-detail-header h4 {{ font-size: 15px; color: #2C2C2E; font-weight: 600; }}
.step-detail-close {{ cursor: pointer; color: #AEAEB2; font-size: 18px; padding: 4px 8px; border-radius: 6px; }}
.step-detail-close:hover {{ color: #2C2C2E; background: rgba(0,0,0,0.04); }}
.step-detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
@media (max-width: 700px) {{ .step-detail-grid {{ grid-template-columns: 1fr; }} }}
.sd-card {{ background: #F9F8F6; border-radius: 10px; padding: 12px; border: 1px solid #ECEAE6; }}
.sd-card h5 {{ font-size: 11px; color: #8E8E93; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; font-weight: 600; }}
.sd-metric {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; border-bottom: 1px solid #F0EEEA; }}
.sd-metric:last-child {{ border-bottom: none; }}
.sd-metric .sd-label {{ color: #8E8E93; }}
.sd-metric .sd-value {{ color: #2C2C2E; font-weight: 600; }}
.timeline-step.selected {{ transform: scaleY(1.35); z-index: 20; outline: 2px solid #7C8B9A; outline-offset: 1px; }}

.detail-panel {{ display: none; margin-top: 12px; padding: 12px; background: rgba(124,139,154,0.06);
                 border-radius: 10px; }}
.example-card.expanded .detail-panel {{ display: block; }}

/* ── Responsive ─────────────────────────────── */
@media (max-width: 900px) {{
    .grid-2, .grid-3, .grid-4, .grid-2-1 {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>

<div class="nav">
    {"<a href='index.html' style='text-decoration:none;color:#8E8E93;font-size:14px;padding:8px 0;'>&#8592; Index</a>" if is_fixed else ""}
    <h1>{("All Models &amp; Datasets" if is_all_page else f"{fixed_model} / {fixed_dataset}") if is_fixed else "HHMM-GMM Reasoning Analysis"}</h1>
    <div class="nav-tabs">
        <button class="nav-tab active" onclick="switchTab('population')">Population</button>
        {individual_tab_btn}
    </div>
</div>

<div class="filter-bar" id="filter-bar" style="display:none;">
    <label>Model</label>
    <select id="filter-model" onchange="onFilterChange()"></select>
    <label>Dataset</label>
    <select id="filter-dataset" onchange="onFilterChange()"></select>
    <span class="filter-status" id="filter-status">All models &amp; datasets (aggregated)</span>
</div>

<div class="container">

<!-- ════════════════════════════════════════════════════════════════ -->
<!--  POPULATION TAB                                                 -->
<!-- ════════════════════════════════════════════════════════════════ -->
<div id="tab-population" class="tab-content active">

    <!-- Transition matrices -->
    <div class="section">
        <div class="grid-2">
            <div>
                <h2>1st-Order Transition Matrices</h2>
                <div class="fn">P(next category | current category). Each cell (i,j) = probability of transitioning from category i to j.</div>
                <div class="btn-group" id="matrix-btns"></div>
                <div id="matrix-plot" style="height:450px;"></div>
            </div>
            <div id="matrix2nd-wrapper" style="display:none;">
                <h2>2nd-Order Transition Matrices</h2>
                <div class="fn">P(next | prev, current). Rows = context pairs (prev→cur), columns = next category.</div>
                <div class="btn-group" id="matrix2nd-btns"></div>
                <div id="matrix2nd-plot" style="height:450px;"></div>
            </div>
        </div>
    </div>

    <!-- Difference heatmaps -->
    <div class="section">
        <h2>Transition Differences</h2>
        <div class="fn">Element-wise difference between group transition matrices.</div>
        <div class="grid-3">
            <div id="diff-corr-incorr" style="height:380px;"></div>
            <div id="diff-corr-long" style="height:380px;"></div>
            <div id="diff-corr-short" style="height:380px;"></div>
        </div>
    </div>

    <!-- Transition flow + Start probs (side by side) -->
    <div class="grid-2-1">
        <div class="section">
            <h2>Transition Flow</h2>
            <div class="fn">Weighted flow between categories. Link thickness = transition probability &times; source weight.</div>
            <div class="btn-group" id="sankey-btns"></div>
            <div id="sankey-plot" style="height:420px;"></div>
        </div>
        <div class="section section-compact">
            <h2>Start Distribution</h2>
            <div class="fn">Probability of each category being the first step in a sequence.</div>
            <div id="start-plot" style="height:280px;"></div>
        </div>
    </div>

    <!-- Path stats + Error split + Stationary + Hitting times in compact row -->
    <div class="grid-4">
        <div class="section section-compact">
            <h2>Path Length</h2>
            <div class="fn">Number of top-level category steps per sequence.</div>
            <div id="path-stats-plot" style="height:260px;"></div>
        </div>
        <div class="section section-compact">
            <h2>Error Split</h2>
            <div class="fn">Incorrect sequences split by path length: long (&ge; 100 steps) vs short (&lt; 100 steps).</div>
            <div id="error-split-plot" style="height:260px;"></div>
        </div>
        <div class="section section-compact">
            <h2>Stationary Dist.</h2>
            <div class="fn">Long-run proportion of time spent in each category (Markov chain &pi;).</div>
            <div id="stationary-plot" style="height:260px;"></div>
        </div>
        <div class="section section-compact">
            <h2>Expected Steps to FA</h2>
            <div class="fn">Expected number of transitions to reach Final Answer from each starting category.</div>
            <div id="hitting-plot" style="height:260px;"></div>
        </div>
    </div>

    <!-- Bottom-level metrics -->
    <div class="section" id="bottom-section" style="display:none;">
        <h2>Bottom-Level Metrics</h2>
        <div class="fn">Hidden-state regime analysis from the bottom GMMs. Regimes = mixture components learned within each category (jointly trained on all data).</div>
        <div class="grid-2">
            <div>
                <div id="pca-scatter-plot" style="height:450px;"></div>
                <div class="fn">2D PCA projection of per-step activations colored by category. Diamond markers = category centroids. Separation between clusters indicates distinct hidden-state representations per category.</div>
            </div>
            <div>
                <div id="td-cosine-plot" style="height:450px;"></div>
                <div class="fn">Cosine similarity between mean transition direction vectors. Each direction = average activation displacement when the model transitions from category A to B. High cosine between two directions means the hidden-state movement is similar.</div>
            </div>
        </div>
        <div style="margin-top:16px;">
            <div id="dir-correctness-plot" style="height:300px;"></div>
            <div class="fn">Cosine similarity between the mean transition direction vector of <b>correct</b> vs <b>incorrect</b> sequences. cos&asymp;1 = both groups move in the same activation direction at this transition; cos&asymp;0 = orthogonal movements; cos&lt;0 = opposite directions. Low or negative cosine indicates a directional signature that distinguishes correct from incorrect reasoning.</div>
        </div>
        <div class="grid-3" style="margin-top:16px;">
            <div>
                <div id="pca-dist-0" style="height:340px;"></div>
                <div class="fn">Density of per-step activations along principal component 0. Separation between category curves indicates the primary axis of hidden-state differentiation.</div>
            </div>
            <div>
                <div id="pca-dist-1" style="height:340px;"></div>
                <div class="fn">Density along principal component 1, the second axis of variation. Captures secondary structure in category representations.</div>
            </div>
            <div>
                <div id="pca-dist-2" style="height:340px;"></div>
                <div class="fn">Density along principal component 2, the third axis of variation.</div>
            </div>
        </div>
        <!-- ── Regime Analysis ──────────────────────────────────────────── -->
        <div style="margin-top:32px; border-top:1px solid #E8E4E0; padding-top:20px;">
            <h3 style="font-size:1.05rem; color:#2C2C2E; margin-bottom:2px;">Regime Analysis</h3>

            <!-- 4. Regime means 2D scatter — PCA -->
            <div id="regime-means-2d-section" style="display:none; margin-top:24px;">
                <h4 style="font-size:0.95rem; color:#3C3C3E; margin-bottom:2px;">Regime Structure</h4>
                <div class="fn">PCA projection (axes capture max variance among regimes). Well-separated islands = regimes are geometrically distinct.</div>
                <div class="grid-2" style="margin-top:10px;">
                    <div>
                        <div style="font-size:12px; font-weight:600; color:#6B6B70; text-align:center; margin-bottom:4px;">FA &mdash; PCA</div>
                        <div id="regime-pca-FA" style="height:300px;"></div>
                    </div>
                    <div>
                        <div style="font-size:12px; font-weight:600; color:#6B6B70; text-align:center; margin-bottom:4px;">SR &mdash; PCA</div>
                        <div id="regime-pca-SR" style="height:300px;"></div>
                    </div>
                </div>
                <div class="grid-2" style="margin-top:8px;">
                    <div>
                        <div style="font-size:12px; font-weight:600; color:#6B6B70; text-align:center; margin-bottom:4px;">AC &mdash; PCA</div>
                        <div id="regime-pca-AC" style="height:300px;"></div>
                    </div>
                    <div>
                        <div style="font-size:12px; font-weight:600; color:#6B6B70; text-align:center; margin-bottom:4px;">UV &mdash; PCA</div>
                        <div id="regime-pca-UV" style="height:300px;"></div>
                    </div>
                </div>
            </div>

            <!-- 4b. Soft Activation Profiles: layer×regime heatmap correct vs incorrect -->
            <div id="soft-profiles-section" style="display:none; margin-top:28px;">
                <h4 style="font-size:0.95rem; color:#3C3C3E; margin-bottom:2px;">Soft Activation Profiles (Correct vs Incorrect)</h4>
                <div class="fn">Mean posterior probability &gamma;(layer, regime) averaged across all steps. Left = correct, middle = incorrect, right = difference (blue = more in correct, red = more in incorrect). Rows = transformer layers (top=early, bottom=late), columns = regimes. Strong colored cells in the difference plot indicate (layer, regime) pairs that discriminate correct from incorrect reasoning.</div>
                <div id="soft-profiles-grid" style="margin-top:10px;"></div>
            </div>

            <!-- 5. Step trajectory -->
            <div id="step-trajectory-section" style="display:none; margin-top:32px;">
                <h4 style="font-size:0.95rem; color:#3C3C3E; margin-bottom:2px;">Step Trajectory: Layer-by-Layer Computation Pattern</h4>
                <div class="fn">Each dot = one transformer layer in training-PCA space (dim 0 vs dim 1). <b>Color &amp; label = regime (MAP-decoded).</b> ◆ = regime transition layer. ✦ faded star = centroid. Open circle = layer 0. By default, 2 random correct and 2 random incorrect samples are shown. Click a case chip to highlight an individual sample.</div>
                <div id="traj-cat-tabs" style="display:flex; gap:6px; margin-top:12px; flex-wrap:wrap;"></div>
                <div style="display:flex; gap:0; margin-top:14px; align-items:stretch; border:1px solid #E8E4E0; border-radius:12px; overflow:hidden; background:#FFFFFF;">
                    <div id="traj-sample-panel" style="width:184px; flex-shrink:0; padding:14px 10px; border-right:1px solid #E8E4E0; max-height:520px; overflow-y:auto; background:#FAF7F4;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-size:10px; color:#8E8E93; font-weight:700; letter-spacing:0.05em;">CASES</span>
                            <button id="traj-show-all-btn" style="font-size:10px; color:#7C8B9A; background:none; border:none; cursor:pointer; padding:2px 6px; border-radius:4px; border:1px solid transparent;">show all</button>
                        </div>
                        <div id="traj-correct-list"></div>
                        <div id="traj-incorrect-list" style="margin-top:4px;"></div>
                    </div>
                    <div style="flex:1; min-width:0;">
                        <div id="traj-plot" style="height:520px;"></div>
                    </div>
                </div>
                <!-- Combined: all categories in one PCA space -->
            </div>

        </div>
    </div>




    <div class="section" id="regime-top-section" style="display:none;">
        <h2>Explicit Bridge</h2>
        <div class="fn">P(target category | source category, exit regime).</div>
        <div id="regime-top-grid" style="margin-top:8px;"></div>
    </div>

</div>

<!-- ════════════════════════════════════════════════════════════════ -->
<!--  INDIVIDUAL TAB                                                 -->
<!-- ════════════════════════════════════════════════════════════════ -->
<div id="tab-individual" class="tab-content">

    <div class="fn" style="margin-bottom:16px; line-height:2.0;">
        <b>Length</b> = number of top-level category steps. Click a card to expand and see the timeline. Click timeline steps for hidden-state details.
    </div>

    <div class="section">
        <h2>Examples</h2>
        <div class="fn">Randomly sampled sequences from all groups. Click to expand. Click timeline steps for details.</div>
        <div id="examples-container"></div>
    </div>

</div>

</div><!-- /container -->

<script>
// ══════════════════════════════════════════════════════════════════
//  DATA
// ══════════════════════════════════════════════════════════════════
const TOP = {top_json};
const BOTTOM = {bottom_json};
const COLORS = {cat_colors_json};
const NAMES = {cat_names_json};
const TAGS = {tags_json};
const C = TAGS.length;

const plotBg = '#FAFAF8';
const paperBg = 'rgba(0,0,0,0)';
const gridColor = '#ECEAE6';
const fontColor = '#2C2C2E';
const plotLayout = {{
    paper_bgcolor: paperBg, plot_bgcolor: plotBg,
    font: {{ color: fontColor, family: '-apple-system, SF Pro Display, Helvetica Neue, sans-serif', size: 12 }},
    margin: {{ t: 45, b: 45, l: 55, r: 25 }},
    xaxis: {{ gridcolor: gridColor, linecolor: '#D6D3CE', zerolinecolor: '#D6D3CE' }},
    yaxis: {{ gridcolor: gridColor, linecolor: '#D6D3CE', zerolinecolor: '#D6D3CE' }},
}};
// Morandi semantic colors
const mGreen = '#5B8DB8';  // success / correct (blue, colorblind-safe)
const mRose = '#D4885C';   // error / incorrect (orange, colorblind-safe)
const mGold = '#C4B176';   // warning / caution
const mTaupe = '#B5A89A';  // neutral
const mSteel = '#7C8B9A';  // accent
const mPlum = '#8B7DAA';   // special / layer
// Morandi colorscales for heatmaps
const morandiSeq = [[0, '#F5F0EB'], [0.25, '#E0C0A0'], [0.5, '#D4885C'], [0.75, '#B5736A'], [1, '#8B4A3A']];
const morandiDiv = [[0, '#D4885C'], [0.25, '#E0C0A0'], [0.5, '#F5F0EB'], [0.75, '#B0C8D8'], [1, '#5B8DB8']];
const morandiDivR = [[0, '#5B8DB8'], [0.25, '#B0C8D8'], [0.5, '#F5F0EB'], [0.75, '#E0C0A0'], [1, '#D4885C']];

// ══════════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════════════════
function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
    currentTab = name;
    if (_tabDirty[name]) {{ _renderTab(name); _tabDirty[name] = false; }}
    window.dispatchEvent(new Event('resize'));
}}

// ══════════════════════════════════════════════════════════════════
//  MODEL / DATASET FILTER
// ══════════════════════════════════════════════════════════════════
let currentModel = {("'all'" if is_all_page else f"'{fixed_model}'") if is_fixed else "'all'"};
let currentDataset = {("'all'" if is_all_page else f"'{fixed_dataset}'") if is_fixed else "'all'"};
let currentTab = 'population';
let _tabDirty = {{ population: true, individual: true }};

function getMdKey() {{
    if (currentModel === 'all' || currentDataset === 'all') return null;
    return currentModel + '/' + currentDataset;
}}

function getTopMd() {{
    const key = getMdKey();
    if (!key || !TOP.per_model_dataset) return null;
    return TOP.per_model_dataset[key] || null;
}}

function getBottomMd() {{
    const key = getMdKey();
    if (!key || !BOTTOM || !BOTTOM.per_model_dataset) return null;
    return BOTTOM.per_model_dataset[key] || null;
}}

function isFiltered() {{ return getMdKey() !== null; }}

function buildFilterBar() {{
    const bar = document.getElementById('filter-bar');
    if (!bar) return;
    const models = TOP.models || [];
    const datasets = TOP.datasets || [];
    if (models.length <= 1 && datasets.length <= 1) {{ bar.style.display = 'none'; return; }}

    const modelSel = document.getElementById('filter-model');
    modelSel.innerHTML = '<option value="all">All</option>' +
        models.map(m => `<option value="${{m}}">${{m}}</option>`).join('');
    updateDatasetOptions();
    bar.style.display = '';
}}

function updateDatasetOptions() {{
    const datasetSel = document.getElementById('filter-dataset');
    let datasets = (TOP.datasets || []).slice();
    if (currentModel !== 'all' && TOP.per_model_dataset) {{
        datasets = datasets.filter(d => TOP.per_model_dataset[currentModel + '/' + d]);
    }}
    datasetSel.innerHTML = '<option value="all">All</option>' +
        datasets.map(d => `<option value="${{d}}">${{d}}</option>`).join('');
    if (currentDataset !== 'all' && !datasets.includes(currentDataset)) currentDataset = 'all';
    datasetSel.value = currentDataset;
}}

function onFilterChange() {{
    currentModel = document.getElementById('filter-model').value;
    updateDatasetOptions();
    currentDataset = document.getElementById('filter-dataset').value;
    updateFilterStatus();
    renderAll();
}}

function updateFilterStatus() {{
    const el = document.getElementById('filter-status');
    if (!el) return;
    const md = getTopMd();
    if (md) {{
        const seeds = md.seeds || [];
        el.textContent = currentModel + ' / ' + currentDataset + ' (' + seeds.length + ' seed' + (seeds.length !== 1 ? 's' : '') + ')';
        el.style.color = '#3A6B96';
    }} else if (currentModel !== 'all' || currentDataset !== 'all') {{
        el.textContent = 'No data for this combination';
        el.style.color = '#C44E52';
    }} else {{
        el.textContent = 'All models & datasets (aggregated)';
        el.style.color = '#8E8E93';
    }}
}}

function aggNote() {{
    return isFiltered() ? '<div class="agg-note">Aggregated data — not available per model/dataset</div>' : '';
}}

function renderAll() {{
    // Clear dynamic containers
    ['soft-profiles-grid', 'regime-top-grid',
     'traj-cat-tabs', 'examples-container'].forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    }});
    // Remove stale aggregate notes
    document.querySelectorAll('.agg-note').forEach(n => n.remove());

    // Mark all tabs dirty, render current tab only
    _tabDirty = {{ population: true, individual: true }};
    _renderTab(currentTab);
    _tabDirty[currentTab] = false;
}}

// ── Lazy rendering infrastructure ──────────────────────────────────
let _deferredGeneration = 0;

function _renderTab(name) {{
    if (name === 'population') {{
        // Immediate: above-the-fold sections
        renderMatrixButtons(); showMatrix(currentMatrix);
        renderMatrix2ndButtons();
        if (Object.keys(TOP['2nd_order_matrices'] || {{}}).length > 0) showMatrix2nd(currentMatrix2nd);
        renderDiffHeatmaps();
        renderStartProbs();
        renderSankeyButtons(); showSankey(currentSankey);
        renderPathStats();
        renderMarkov();
        labelDataSources();

        // Deferred: below-the-fold expensive sections.
        // Uses setTimeout to yield to the browser between each section,
        // so the above-the-fold content paints immediately.
        _deferredGeneration++;
        const gen = _deferredGeneration;
        const deferred = [
            renderBottom,
            renderRegimeMeans2D,
            renderSoftProfiles,
            renderStepTrajectory,
            renderRegimeTopTransitions,
        ];
        (function runNext(i) {{
            if (i >= deferred.length || gen !== _deferredGeneration) return;
            setTimeout(() => {{
                if (gen !== _deferredGeneration) return;  // filter changed, abort
                deferred[i]();
                runNext(i + 1);
            }}, 0);
        }})(0);
    }} else if (name === 'individual') {{
        renderExamples();
        labelDataSources();
    }}
}}

// ══════════════════════════════════════════════════════════════════
//  SEED SOURCE LABELS
// ══════════════════════════════════════════════════════════════════
function labelDataSources() {{
    // Build compact seed description from aggregate config
    function seedDesc(runLabels, nRuns) {{
        if (!runLabels && !nRuns) return null;
        const n = runLabels ? runLabels.length : nRuns;
        if (!n) return null;
        return `${{n}} run${{n !== 1 ? 's' : ''}}`;
    }}

    // For per-md pages, use seed count from per-md data; for all page, use global config
    const mdTop = getTopMd();
    const mdSeeds = mdTop && mdTop.seeds;
    const perMdN = mdSeeds ? mdSeeds.length : null;
    const topRuns = TOP && TOP.config && TOP.config.run_labels;
    const globalDesc = seedDesc(topRuns, TOP && TOP.config && TOP.config.num_runs)
                    || seedDesc(null, BOTTOM && BOTTOM.n_runs);
    const aggDesc = perMdN ? seedDesc(null, perMdN) : globalDesc;

    function badge(desc) {{
        return `<span class="seed-badge" style="font-size:11px; color:#AEAEB2; font-weight:400; margin-left:6px;">${{desc}}</span>`;
    }}
    // Remove old badges before re-applying
    document.querySelectorAll('.seed-badge').forEach(b => b.remove());
    document.querySelectorAll('[data-seed-labeled]').forEach(el => delete el.dataset.seedLabeled);

    function label(el, desc) {{
        if (el && !el.dataset.seedLabeled) {{
            el.innerHTML += badge(desc);
            el.dataset.seedLabeled = '1';
        }}
    }}

    if (!aggDesc) return;

    // All section headings get the same badge
    document.querySelectorAll('#tab-population .section h2').forEach(h2 => label(h2, aggDesc));
    label(document.querySelector('#bottom-section > h2'), aggDesc);
    document.querySelectorAll('#bottom-section h4').forEach(h4 => label(h4, aggDesc));
    document.querySelectorAll('#bottom-section h3').forEach(h3 => label(h3, aggDesc));
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Transition Matrices
// ══════════════════════════════════════════════════════════════════
let currentMatrix = 'all';

function _getMatrices() {{
    const mdTop = getTopMd();
    return (mdTop && mdTop['1st_order_matrices']) || TOP['1st_order_matrices'] || {{}};
}}

function computeStdMat(mats) {{
    return mats[0].map((row, i) => row.map((_, j) => {{
        const vals = mats.map(m => m[i][j]);
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        return Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1));
    }}));
}}

function renderMatrixButtons() {{
    const el = document.getElementById('matrix-btns');
    const names = ['all', 'correct', 'incorrect', 'long_fail', 'short_fail'];
    const labels = ['All', 'Correct', 'Incorrect', 'Long Fail', 'Short Fail'];
    const mats = _getMatrices();
    el.innerHTML = names.filter(n => n in mats).map((n, i) =>
        `<button class="btn ${{n === currentMatrix ? 'active' : ''}}" onclick="showMatrix('${{n}}')">${{labels[i]}}</button>`
    ).join('');
}}

function showMatrix(name) {{
    currentMatrix = name;
    renderMatrixButtons();
    const mats = _getMatrices();
    const src = mats[name];
    if (!src) return;

    // Deep-copy to prevent Plotly from mutating the source data
    const mat = src.map(row => [...row]);

    // Compute std: cross-MD in "all" view, cross-seed in per-md view
    let stdMat = null;
    if (!isFiltered() && TOP.per_model_dataset) {{
        const allMats = Object.values(TOP.per_model_dataset)
            .map(e => (e['1st_order_matrices'] || {{}})[name])
            .filter(Boolean);
        if (allMats.length >= 2) stdMat = computeStdMat(allMats);
    }} else if (isFiltered()) {{
        const mdTop = getTopMd();
        const seedMats = mdTop && mdTop.seed_matrices && mdTop.seed_matrices[name];
        if (seedMats && seedMats.length >= 2) stdMat = computeStdMat(seedMats);
    }}

    const text = mat.map((row, i) => row.map((v, j) =>
        stdMat ? v.toFixed(3) + '<br>(\u00b1' + stdMat[i][j].toFixed(3) + ')' : v.toFixed(3)
    ));
    const trace = {{
        z: mat, x: TAGS, y: TAGS, type: 'heatmap',
        colorscale: morandiDiv,
        text: text, texttemplate: '%{{text}}', textfont: {{ size: stdMat ? 11 : 14 }},
        hovertemplate: 'From: %{{y}}<br>To: %{{x}}<br>P = %{{z:.4f}}<extra></extra>',
    }};
    const titles = {{ all: 'All', correct: 'Correct', incorrect: 'Incorrect',
                      long_fail: 'Long Failures', short_fail: 'Short Failures' }};
    const layout = {{
        ...plotLayout, title: {{ text: '1st-Order: ' + (titles[name] || name), font: {{ size: 15, color: fontColor, family: plotLayout.font.family }} }},
        xaxis: {{ ...plotLayout.xaxis, title: 'To', side: 'bottom' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'From', autorange: 'reversed' }},
    }};
    Plotly.react('matrix-plot', [trace], layout, {{ responsive: true }});
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: 2nd-Order Transition Matrices
// ══════════════════════════════════════════════════════════════════
let currentMatrix2nd = 'all';
const CTX_LABELS = [];
for (let i = 0; i < C; i++) for (let j = 0; j < C; j++) CTX_LABELS.push(TAGS[i] + '→' + TAGS[j]);

function renderMatrix2ndButtons() {{
    const mats2 = TOP['2nd_order_matrices'] || {{}};
    if (Object.keys(mats2).length === 0) {{ document.getElementById('matrix2nd-wrapper').style.display = 'none'; return; }}
    document.getElementById('matrix2nd-wrapper').style.display = '';
    const el = document.getElementById('matrix2nd-btns');
    const names = ['all', 'correct', 'incorrect'];
    const labels = ['All', 'Correct', 'Incorrect'];
    el.innerHTML = names.filter(n => n in mats2).map((n, i) =>
        `<button class="btn ${{n === currentMatrix2nd ? 'active' : ''}}" onclick="showMatrix2nd('${{n}}')">${{labels[names.indexOf(n)]}}</button>`
    ).join('');
}}

function showMatrix2nd(name) {{
    currentMatrix2nd = name;
    renderMatrix2ndButtons();
    const mats2 = TOP['2nd_order_matrices'] || {{}};
    const src = mats2[name];
    if (!src) return;

    // src is (C*C)×C — rows = context pairs, cols = next category
    const mat = src.map(row => [...row]);
    const text = mat.map(row => row.map(v => v.toFixed(3)));
    const trace = {{
        z: mat, x: TAGS, y: CTX_LABELS, type: 'heatmap',
        colorscale: morandiDiv,
        text: text, texttemplate: '%{{text}}', textfont: {{ size: 11 }},
        hovertemplate: 'Context: %{{y}}<br>Next: %{{x}}<br>P = %{{z:.4f}}<extra></extra>',
    }};
    const titles = {{ all: 'All', correct: 'Correct', incorrect: 'Incorrect' }};
    const layout = {{
        ...plotLayout,
        title: {{ text: '2nd-Order: ' + (titles[name] || name), font: {{ size: 15, color: fontColor, family: plotLayout.font.family }} }},
        xaxis: {{ ...plotLayout.xaxis, title: 'Next', side: 'bottom' }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Context (prev→cur)', autorange: 'reversed', tickfont: {{ size: 10 }} }},
        margin: {{ ...plotLayout.margin, l: 80 }},
    }};
    Plotly.react('matrix2nd-plot', [trace], layout, {{ responsive: true }});
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Difference Heatmaps
// ══════════════════════════════════════════════════════════════════
function renderDiffHeatmaps() {{
    const mats = _getMatrices();

    function plotDiff(divId, nameA, nameB, title) {{
        if (!(nameA in mats) || !(nameB in mats)) {{
            document.getElementById(divId).innerHTML = '<p style="color:#AEAEB2;text-align:center;padding:40px;">Insufficient data</p>';
            return;
        }}
        const a = mats[nameA], b = mats[nameB];
        const diff = a.map((row, i) => row.map((v, j) => v - b[i][j]));
        const flat = diff.flat();
        const vlim = Math.max(Math.abs(Math.min(...flat)), Math.abs(Math.max(...flat)), 0.01);

        // Compute std of diff: cross-MD in "all" view, cross-seed in per-md view
        let diffStdMat = null;
        if (!isFiltered() && TOP.per_model_dataset) {{
            const diffMats = Object.values(TOP.per_model_dataset).map(e => {{
                const ma = (e['1st_order_matrices'] || {{}})[nameA];
                const mb = (e['1st_order_matrices'] || {{}})[nameB];
                return (ma && mb) ? ma.map((row, i) => row.map((v, j) => v - mb[i][j])) : null;
            }}).filter(Boolean);
            if (diffMats.length >= 2) diffStdMat = computeStdMat(diffMats);
        }} else if (isFiltered()) {{
            const mdTop = getTopMd();
            const smA = mdTop && mdTop.seed_matrices && mdTop.seed_matrices[nameA];
            const smB = mdTop && mdTop.seed_matrices && mdTop.seed_matrices[nameB];
            if (smA && smB && smA.length >= 2) {{
                const diffMats = smA.map((ma, k) => ma.map((row, i) => row.map((v, j) => v - smB[k][i][j])));
                diffStdMat = computeStdMat(diffMats);
            }}
        }}

        const text = diff.map((row, i) => row.map((v, j) => {{
            const s = v >= 0 ? '+' : '';
            return diffStdMat
                ? s + v.toFixed(3) + '<br>(\u00b1' + diffStdMat[i][j].toFixed(3) + ')'
                : s + v.toFixed(3);
        }}));
        const trace = {{
            z: diff, x: TAGS, y: TAGS, type: 'heatmap',
            colorscale: morandiDiv, zmid: 0, zmin: -vlim, zmax: vlim,
            text: text, texttemplate: '%{{text}}', textfont: {{ size: diffStdMat ? 10 : 13 }},
            hovertemplate: 'From: %{{y}}<br>To: %{{x}}<br>\u0394P = %{{z:+.4f}}<extra></extra>',
        }};
        const layout = {{
            ...plotLayout, title: {{ text: title, font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
            xaxis: {{ ...plotLayout.xaxis, title: 'To' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'From', autorange: 'reversed' }},
        }};
        Plotly.newPlot(divId, [trace], layout, {{ responsive: true }});
    }}

    plotDiff('diff-corr-incorr', 'correct', 'incorrect', 'Correct \u2212 Incorrect');
    plotDiff('diff-corr-long', 'correct', 'long_fail', 'Correct \u2212 Long Fail');
    plotDiff('diff-corr-short', 'correct', 'short_fail', 'Correct \u2212 Short Fail');
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Start Probs
// ══════════════════════════════════════════════════════════════════
function renderStartProbs() {{
    const mdTop = getTopMd();
    // per_model_dataset has {{FA:{{mean,std}},...}}; aggregate has nested {{all:{{FA:x,...}},...}}
    const sp = (mdTop && mdTop.start_probs) ? mdTop.start_probs : (TOP.start_probs || {{}}).all;
    if (!sp) return;
    // Support both {{tag: number}} and {{tag: {{mean, std}}}} formats
    const getVal = v => (v != null && typeof v === 'object') ? v.mean : (v || 0);
    const getStd = v => (v != null && typeof v === 'object') ? (v.std || 0) : 0;
    const vals = TAGS.map(t => getVal(sp[t]));
    const stds = TAGS.map(t => getStd(sp[t]));
    const hasErr = stds.some(s => s > 0);
    const colors = TAGS.map(t => COLORS[t]);
    const trace = {{
        x: TAGS, y: vals, type: 'bar',
        marker: {{ color: colors, line: {{ color: '#E8E4E0', width: 1 }} }},
        text: vals.map((v, i) => hasErr && stds[i] ? v.toFixed(3) + '\u00b1' + stds[i].toFixed(3) : v.toFixed(3)),
        textposition: 'outside',
        ...(hasErr ? {{ error_y: {{ type: 'data', array: stds, visible: true, color: '#AEAEB2' }},
            hovertemplate: '%{{x}}: %{{y:.4f}} \u00b1 %{{error_y.array:.4f}}<extra></extra>' }}
            : {{ hovertemplate: '%{{x}}: %{{y:.4f}}<extra></extra>' }}),
    }};
    const layout = {{
        ...plotLayout,
        title: {{ text: 'Start Category Distribution', font: {{ size: 15, color: fontColor, family: plotLayout.font.family }} }},
        yaxis: {{ ...plotLayout.yaxis, title: 'Probability', range: [0, Math.max(...vals.map((v,i) => v + (stds[i]||0))) * 1.2] }},
    }};
    Plotly.newPlot('start-plot', [trace], layout, {{ responsive: true }});
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Sankey
// ══════════════════════════════════════════════════════════════════
let currentSankey = 'all';

function renderSankeyButtons() {{
    const el = document.getElementById('sankey-btns');
    const names = ['all', 'correct', 'incorrect'];
    const labels = ['All', 'Correct', 'Incorrect'];
    const mats = _getMatrices();
    el.innerHTML = names.filter(n => n in mats).map(n =>
        `<button class="btn ${{n === currentSankey ? 'active' : ''}}" onclick="showSankey('${{n}}')">${{labels[names.indexOf(n)]}}</button>`
    ).join('');
}}

function showSankey(name) {{
    currentSankey = name;
    renderSankeyButtons();
    const mat = _getMatrices()[name];
    if (!mat) return;

    // Nodes: from-side (0..3) and to-side (4..7), fixed positions
    const nodeLabels = TAGS.map(t => t).concat(TAGS.map(t => t));
    const nodeColors = TAGS.map(t => COLORS[t]).concat(TAGS.map(t => COLORS[t]));
    // Fix node positions: from-nodes at x=0.01, to-nodes at x=0.99
    // y evenly spaced: FA=0.1, SR=0.35, AC=0.6, UV=0.85
    const yPos = [0.01, 0.3, 0.6, 0.9];
    const nodeX = Array(C).fill(0.01).concat(Array(C).fill(0.99));
    const nodeY = yPos.concat(yPos);

    const source = [], target = [], value = [], linkColor = [], linkLabels = [];

    for (let i = 0; i < C; i++) {{
        for (let j = 0; j < C; j++) {{
            if (mat[i][j] > 0.005) {{
                source.push(i);
                target.push(C + j);
                value.push(mat[i][j] * 100);
                const hex = COLORS[TAGS[i]];
                linkColor.push(hex + '66');
                linkLabels.push(TAGS[i] + ' → ' + TAGS[j] + ': ' + (mat[i][j]*100).toFixed(1) + '%');
            }}
        }}
    }}

    const trace = {{
        type: 'sankey',
        arrangement: 'fixed',
        valueformat: '.1f', valuesuffix: '%',
        node: {{ label: nodeLabels, color: nodeColors, pad: 35, thickness: 20,
                 x: nodeX, y: nodeY,
                 line: {{ color: '#E8E4E0', width: 0.5 }},
                 hovertemplate: '%{{label}}<extra></extra>' }},
        link: {{ source, target, value, color: linkColor, label: linkLabels,
                 hovertemplate: '%{{label}}<extra></extra>' }},
    }};
    const titles = {{ all: 'All Sequences', correct: 'Correct Only', incorrect: 'Incorrect Only' }};
    const layout = {{
        ...plotLayout,
        title: {{ text: 'Transition Flow: ' + (titles[name] || name), font: {{ size: 15, color: fontColor, family: plotLayout.font.family }} }},
    }};
    Plotly.newPlot('sankey-plot', [trace], layout, {{ responsive: true }});
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Path & Error Stats
// ══════════════════════════════════════════════════════════════════
function renderPathStats() {{
    const mdTop = getTopMd();
    const pop = (mdTop && mdTop.population_stats) || TOP.population_stats || {{}};

    // Path length comparison (correct vs incorrect)
    const cpl = pop.correct_path_length_mean;
    const ipl = pop.incorrect_path_length_mean;
    if (cpl && ipl) {{
        const trace = {{
            x: ['Correct', 'Incorrect'], y: [cpl.mean, ipl.mean], type: 'bar',
            marker: {{ color: [mGreen, mRose], line: {{ color: '#E8E4E0', width: 1 }} }},
            error_y: {{ type: 'data', array: [cpl.std, ipl.std], visible: true, color: '#AEAEB2' }},
            text: [cpl.mean.toFixed(1), ipl.mean.toFixed(1)], textposition: 'outside',
            hovertemplate: '%{{x}}: %{{y:.1f}} \u00b1 %{{error_y.array:.1f}}<extra></extra>',
        }};
        const layout = {{
            ...plotLayout,
            title: {{ text: 'Avg Path Length', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
            yaxis: {{ ...plotLayout.yaxis, title: 'Sentence Labels' }},
        }};
        Plotly.newPlot('path-stats-plot', [trace], layout, {{ responsive: true }});
    }}

    // Error split
    const lf = pop.long_failures, sf = pop.short_failures;
    if (lf && sf) {{
        const trace = {{
            labels: ['Long Failures', 'Short Failures'],
            values: [lf.mean, sf.mean], type: 'pie',
            marker: {{ colors: [mRose, mGold] }},
            textinfo: 'label+percent', textfont: {{ color: '#fff' }},
            hovertemplate: '%{{label}}: %{{value:.0f}} avg<extra></extra>',
        }};
        const layout = {{
            ...plotLayout,
            title: {{ text: 'Error Split', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
            showlegend: false,
        }};
        Plotly.newPlot('error-split-plot', [trace], layout, {{ responsive: true }});
    }}
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Markov Properties
// ══════════════════════════════════════════════════════════════════
function _crossMdStd(keyFn) {{
    // Compute per-tag std across all per_model_dataset entries using keyFn(entry) -> {{tag: value}}.
    if (!TOP.per_model_dataset) return {{}};
    const entries = Object.values(TOP.per_model_dataset).map(keyFn).filter(Boolean);
    if (entries.length < 2) return {{}};
    const tags = Object.keys(entries[0]);
    const result = {{}};
    tags.forEach(t => {{
        const raw = entries.map(e => e[t]).filter(v => v != null);
        const vals = raw.map(v => (typeof v === 'object') ? v.mean : v).filter(v => v != null);
        if (vals.length < 2) {{ result[t] = 0; return; }}
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        result[t] = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (vals.length - 1));
    }});
    return result;
}}

function renderMarkov() {{
    const mdTop = getTopMd();
    const inAll = !isFiltered();

    // Stationary distribution
    const sd = (mdTop && mdTop.stationary_distribution) || TOP.stationary_distribution;
    if (sd) {{
        const getVal = v => (v != null && typeof v === 'object') ? v.mean : (v || 0);
        const getStdV = v => (v != null && typeof v === 'object') ? (v.std || 0) : 0;
        const vals = TAGS.map(t => getVal(sd[t]));
        const sdStd = inAll ? _crossMdStd(e => e.stationary_distribution) : {{}};
        const errVals = TAGS.map(t => getStdV(sd[t]) || sdStd[t] || null);
        const hasErr = errVals.some(v => v != null && v > 0);
        const trace = {{
            x: TAGS, y: vals, type: 'bar',
            marker: {{ color: TAGS.map(t => COLORS[t]), line: {{ color: '#E8E4E0', width: 1 }} }},
            text: vals.map((v, i) => hasErr && errVals[i] ? v.toFixed(3) + '\u00b1' + errVals[i].toFixed(3) : v.toFixed(3)),
            textposition: 'outside',
            ...(hasErr ? {{ error_y: {{ type: 'data', array: errVals, visible: true, color: '#AEAEB2' }},
                hovertemplate: '%{{x}}: %{{y:.3f}} \u00b1 %{{error_y.array:.3f}}<extra></extra>' }} : {{}}),
        }};
        const layout = {{
            ...plotLayout,
            title: {{ text: 'Stationary Distribution', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
            yaxis: {{ ...plotLayout.yaxis, title: '\u03c0', range: [0, Math.max(...vals.map((v,i) => v + (errVals[i]||0)))*1.25] }},
        }};
        Plotly.newPlot('stationary-plot', [trace], layout, {{ responsive: true }});
    }}

    // Hitting times
    const ht = (mdTop && mdTop.hitting_times_to_FA) || TOP.hitting_times_to_FA;
    if (ht) {{
        const getHtVal = v => (v != null && typeof v === 'object') ? v.mean : v;
        const getHtStd = v => (v != null && typeof v === 'object') ? (v.std || 0) : 0;
        const cats = Object.keys(ht).filter(k => k !== 'final_answer' && k !== 'FA' && k !== 'unknown' && k !== 'UN');
        const displayNames = cats.map(k => {{
            const idx = ['final_answer','setup_and_retrieval','analysis_and_computation','uncertainty_and_verification'].indexOf(k);
            return idx >= 0 ? TAGS[idx] : k;
        }});
        const vals = cats.map(k => getHtVal(ht[k]));
        const htStd = inAll ? _crossMdStd(e => e.hitting_times_to_FA) : {{}};
        const errHt = cats.map(k => getHtStd(ht[k]) || htStd[k] || null);
        const hasErrHt = errHt.some(v => v != null && v > 0);
        const trace = {{
            x: displayNames, y: vals, type: 'bar',
            marker: {{ color: mSteel, line: {{ color: '#E8E4E0', width: 1 }} }},
            text: vals.map((v, i) => v == null ? '—' : (hasErrHt && errHt[i] ? v.toFixed(1) + '\u00b1' + errHt[i].toFixed(1) : v.toFixed(1))),
            textposition: 'outside',
            ...(hasErrHt ? {{ error_y: {{ type: 'data', array: errHt, visible: true, color: '#AEAEB2' }},
                hovertemplate: '%{{x}}: %{{y:.1f}} \u00b1 %{{error_y.array:.1f}}<extra></extra>' }} : {{}}),
        }};
        const layout = {{
            ...plotLayout,
            margin: {{ ...plotLayout.margin, t: 55 }},
            title: {{ text: 'Expected Steps to Final Answer', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
            yaxis: {{ ...plotLayout.yaxis, title: 'Steps', range: [0, Math.max(...vals.map((v,i) => (v||0) + (errHt[i]||0))) * 1.2] }},
        }};
        Plotly.newPlot('hitting-plot', [trace], layout, {{ responsive: true }});
    }}
}}




// ══════════════════════════════════════════════════════════════════
//  Explicit Bridge P(c2 | c1, exit_regime=k)
// ══════════════════════════════════════════════════════════════════
function renderRegimeTopTransitions() {{
    if (!isFiltered()) {{ document.getElementById('regime-top-section').style.display = 'none'; return; }}
    const mdBot = getBottomMd();
    const rtt = (mdBot && mdBot.regime_top_transitions) || (BOTTOM && BOTTOM.regime_top_transitions);
    const grid = document.getElementById('regime-top-grid');
    if (!grid) return;
    if (!rtt || !rtt.per_regime || rtt.per_regime.length === 0) {{
        document.getElementById('regime-top-section').style.display = 'none';
        return;
    }}
    document.getElementById('regime-top-section').style.display = '';

    const K = rtt.K;
    const cats = ['FA', 'SR', 'AC', 'UV'];
    const perRegime = rtt.per_regime;
    const morandiSeq = [[0,'#F5F0EB'],[0.25,'#D0D8CC'],[0.5,'#96B09E'],[0.75,'#5A7A6A'],[1,'#3A5A4A']];

    // One heatmap per source category: rows = regimes, columns = target categories
    cats.forEach(c1 => {{
        if (!perRegime[0] || !perRegime[0][c1]) return;

        // Build K×4 matrix: P(c2 | c1, exit_regime=k)
        const mat = [];
        for (let k = 0; k < K; k++) {{
            const row = [];
            const rd = perRegime[k][c1];
            if (!rd) {{ mat.push(cats.map(() => 0)); return; }}
            cats.forEach(c2 => {{
                row.push(rd.dist[c2] || 0);
            }});
            mat.push(row);
        }}

        const regimeTicks = Array.from({{length: K}}, (_, i) => 'R' + i);
        const fs = K <= 6 ? 10 : 8;

        // Use a shared flex row; create it on the first of every pair
        if (!grid._rttRow || grid._rttRowCount >= 2) {{
            grid._rttRow = document.createElement('div');
            grid._rttRow.style.cssText = 'display:flex; gap:16px; margin-bottom:16px;';
            grid.appendChild(grid._rttRow);
            grid._rttRowCount = 0;
        }}
        const wrapDiv = document.createElement('div');
        wrapDiv.style.cssText = 'width:calc(50% - 8px); min-width:0;';
        wrapDiv.innerHTML = `
            <h3 style="font-size:13px; color:#6B6B70; margin-bottom:2px;">
                Source: ${{c1}}
            </h3>
            <div style="font-size:11px; font-weight:600; color:#6B6B70; margin-bottom:2px; text-align:center;">P(target | source=${{c1}}, exit regime)</div>
            <div id="rtt-${{c1}}-abs" style="height:${{Math.max(180, K * 30 + 60)}}px;"></div>`;
        grid._rttRow.appendChild(wrapDiv);
        grid._rttRowCount++;

        const baseLayout = {{
            ...plotLayout,
            margin: {{ t: 6, b: 48, l: 80, r: 8 }},
            xaxis: {{ ...plotLayout.xaxis, title: {{ text: 'target category', font: {{ size: 10 }} }}, tickfont: {{ size: 10 }} }},
            yaxis: {{ ...plotLayout.yaxis, title: '', autorange: 'reversed', tickfont: {{ size: 9 }} }},
        }};

        // Absolute heatmap: P(c2 | c1, regime=k)
        const text2d = mat.map(row => row.map(v => (v * 100).toFixed(0) + '%'));
        Plotly.newPlot(`rtt-${{c1}}-abs`, [{{
            z: mat, x: cats, y: regimeTicks,
            type: 'heatmap', colorscale: morandiSeq,
            zmin: 0, zmax: Math.max(...mat.flat(), 0.01),
            text: text2d, texttemplate: '%{{text}}',
            textfont: {{ size: fs, color: '#333' }},
            hovertemplate: '%{{y}} &rarr; %{{x}}: %{{z:.3f}}<extra></extra>',
            showscale: false,
        }}], baseLayout, {{ responsive: true }});

    }});
}}

// ══════════════════════════════════════════════════════════════════
//  POPULATION: Bottom-Level Metrics
// ══════════════════════════════════════════════════════════════════
function renderBottom() {{
    if (!isFiltered()) {{ document.getElementById('bottom-section').style.display = 'none'; return; }}
    const mdBot = getBottomMd();
    const m = (mdBot && mdBot.metrics) || (BOTTOM && BOTTOM.metrics);
    if (!m) {{ document.getElementById('bottom-section').style.display = 'none'; return; }}
    document.getElementById('bottom-section').style.display = '';

    // ── Category activation distributions ──
    const catDist = (mdBot && mdBot.category_distributions) || (BOTTOM && BOTTOM.category_distributions);
    if (catDist) {{
        // 2D PCA scatter with confidence ellipses
        const scatterTraces = [];
        TAGS.forEach(t => {{
            const d = catDist[t];
            if (!d) return;
            // Scatter of sampled points
            if (d.pca_samples_0 && d.pca_samples_1) {{
                scatterTraces.push({{
                    x: d.pca_samples_0, y: d.pca_samples_1,
                    mode: 'markers', type: 'scatter', name: t,
                    marker: {{ color: COLORS[t], size: 4, opacity: 0.35 }},
                    hovertemplate: t + '<br>PC0=%{{x:.2f}}<br>PC1=%{{y:.2f}}<extra></extra>',
                }});
            }}
            // Centroid marker
            if (d.mean_0 != null && d.mean_1 != null) {{
                scatterTraces.push({{
                    x: [d.mean_0], y: [d.mean_1],
                    mode: 'markers+text', type: 'scatter', name: t + ' centroid',
                    marker: {{ color: COLORS[t], size: 14, symbol: 'diamond',
                              line: {{ color: '#fff', width: 2 }} }},
                    error_x: (d.min_0 != null && d.max_0 != null) ? {{
                        type: 'data',
                        array: [d.max_0 - d.mean_0],
                        arrayminus: [d.mean_0 - d.min_0],
                        visible: true, color: COLORS[t], thickness: 1.5, width: 4
                    }} : undefined,
                    error_y: (d.min_1 != null && d.max_1 != null) ? {{
                        type: 'data',
                        array: [d.max_1 - d.mean_1],
                        arrayminus: [d.mean_1 - d.min_1],
                        visible: true, color: COLORS[t], thickness: 1.5, width: 4
                    }} : undefined,
                    text: [t], textposition: 'top center',
                    textfont: {{ color: COLORS[t], size: 12, family: plotLayout.font.family }},
                    showlegend: false,
                    hovertemplate: t + ' centroid<br>PC0=%{{x:.2f}} [' + (d.min_0 != null ? d.min_0.toFixed(2) : '?') + ', ' + (d.max_0 != null ? d.max_0.toFixed(2) : '?') + ']<br>PC1=%{{y:.2f}} [' + (d.min_1 != null ? d.min_1.toFixed(2) : '?') + ', ' + (d.max_1 != null ? d.max_1.toFixed(2) : '?') + ']<extra></extra>',
                }});
            }}
        }});
        if (scatterTraces.length > 0) {{
            Plotly.newPlot('pca-scatter-plot', scatterTraces, {{
                ...plotLayout,
                title: {{ text: 'Category Activation Space (PCA)', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
                xaxis: {{ ...plotLayout.xaxis, title: 'PC 0', type: 'linear' }},
                yaxis: {{ ...plotLayout.yaxis, title: 'PC 1', type: 'linear' }},
                legend: {{ font: {{ color: fontColor, size: 11 }}, bgcolor: 'rgba(0,0,0,0)' }},
            }}, {{ responsive: true }});
        }}

        // Per-dimension distribution (overlapping histograms)
        for (let dim = 0; dim < 3; dim++) {{
            const traces = [];
            TAGS.forEach(t => {{
                const d = catDist[t];
                if (!d || !d['hist_edges_' + dim] || !d['hist_counts_' + dim]) return;
                const edges = d['hist_edges_' + dim];
                const counts = d['hist_counts_' + dim];
                // Convert counts to density
                const total = counts.reduce((a, b) => a + b, 0);
                const binW = edges[1] - edges[0];
                const density = counts.map(c => total > 0 ? c / (total * binW) : 0);
                // Use bin centers
                const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
                traces.push({{
                    x: centers, y: density, type: 'scatter', mode: 'lines',
                    name: t, fill: 'tozeroy',
                    line: {{ color: COLORS[t], width: 2 }},
                    fillcolor: COLORS[t] + '33',
                    hovertemplate: t + '<br>PC' + dim + '=%{{x:.2f}}<br>density=%{{y:.3f}}<extra></extra>',
                }});
            }});
            if (traces.length > 0) {{
                Plotly.newPlot('pca-dist-' + dim, traces, {{
                    ...plotLayout,
                    title: {{ text: 'PC ' + dim + ' Distribution', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
                    xaxis: {{ ...plotLayout.xaxis, title: 'PC ' + dim, type: 'linear' }},
                    yaxis: {{ ...plotLayout.yaxis, title: 'Density', type: 'linear' }},
                    legend: {{ font: {{ color: fontColor, size: 11 }}, bgcolor: 'rgba(0,0,0,0)' }},
                }}, {{ responsive: true }});
            }}
        }}
    }}

    // Cross-transition cosine similarity heatmap
    const tdKeys = Object.keys(m).filter(k => k.startsWith('td_mag_')).map(k => k.replace('td_mag_', ''));
    if (tdKeys.length > 0) {{
        const n = tdKeys.length;
        const cosMat = Array.from({{length: n}}, () => Array(n).fill(0));
        for (let i = 0; i < n; i++) cosMat[i][i] = 1;
        // Look for cross_cos_* keys or build from available data
        for (let i = 0; i < n; i++) {{
            for (let j = i + 1; j < n; j++) {{
                const key1 = `td_cross_cos_${{tdKeys[i]}} vs ${{tdKeys[j]}}`;
                const key2 = `td_cross_cos_${{tdKeys[j]}} vs ${{tdKeys[i]}}`;
                const val = m[key1] || m[key2];
                if (val != null) {{
                    const v = typeof val === 'object' ? val.mean : val;
                    cosMat[i][j] = v;
                    cosMat[j][i] = v;
                }}
            }}
        }}
        const tdLabels = tdKeys.map(k => k.replace('->', '→'));
        const text = cosMat.map(row => row.map(v => v.toFixed(2)));
        Plotly.newPlot('td-cosine-plot', [{{
            z: cosMat, x: tdLabels, y: tdLabels, type: 'heatmap',
            colorscale: morandiDivR,
            text, texttemplate: '%{{text}}', textfont: {{ size: 10 }},
            hovertemplate: '%{{y}} vs %{{x}}: cos=%{{z:.3f}}<extra></extra>',
        }}], {{
            ...plotLayout,
            title: {{ text: 'Cross-Transition Direction Cosine', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
            yaxis: {{ ...plotLayout.yaxis, autorange: 'reversed', tickfont: {{ size: 9 }} }},
            xaxis: {{ ...plotLayout.xaxis, tickangle: -45, tickfont: {{ size: 9 }} }},
            margin: {{ ...plotLayout.margin, l: 80, b: 70 }},
        }}, {{ responsive: true }});
    }}

    // ── Direction: correct vs incorrect comparison ──
    const dcc = (mdBot && mdBot.direction_correctness_comparison) || (BOTTOM && BOTTOM.direction_correctness_comparison);
    if (dcc) {{
        const tags = Object.keys(dcc).filter(k => !k.startsWith('_') && dcc[k].dir_cosine != null);
        tags.sort((a, b) => dcc[a].dir_cosine - dcc[b].dir_cosine);
        if (tags.length > 0) {{
            const cosines = tags.map(t => dcc[t].dir_cosine);
            const barColors = cosines.map(c => c >= 0 ? mGreen : mRose);
            const hover = tags.map(t => {{
                const d = dcc[t];
                const magCorr = d.magnitude_correct != null ? d.magnitude_correct.toFixed(3) : '—';
                const magIncorr = d.magnitude_incorrect != null ? d.magnitude_incorrect.toFixed(3) : '—';
                return `${{t.replace('->', '→')}}<br>cos=${{d.dir_cosine.toFixed(3)}}<br>n_correct=${{d.n_correct}}  n_incorrect=${{d.n_incorrect}}<br>mag_correct=${{magCorr}}  mag_incorrect=${{magIncorr}}`;
            }});
            Plotly.newPlot('dir-correctness-plot', [{{
                type: 'bar', orientation: 'h',
                y: tags.map(t => t.replace('->', '→')),
                x: cosines,
                marker: {{ color: barColors }},
                text: cosines.map(c => c.toFixed(2)),
                textposition: 'outside',
                hovertext: hover, hoverinfo: 'text',
            }}], {{
                ...plotLayout,
                title: {{ text: 'Transition Direction Similarity: Correct vs Incorrect', font: {{ size: 14, color: fontColor, family: plotLayout.font.family }} }},
                xaxis: {{ ...plotLayout.xaxis, type: 'linear', autorange: false, range: [0.5, 1.0],
                          tickmode: 'linear', tick0: 0.5, dtick: 0.1,
                          title: 'cos(correct direction, incorrect direction)' }},
                yaxis: {{ ...plotLayout.yaxis, autorange: 'reversed' }},
                margin: {{ ...plotLayout.margin, l: 80 }},
            }}, {{ responsive: true }});
        }}
    }}

}}


// ══════════════════════════════════════════════════════════════════
//  REGIME SANITY CHECKS
// ══════════════════════════════════════════════════════════════════

function renderRegimeMeans2D() {{
    const mdBot = getBottomMd();
    const rc = (mdBot && mdBot.regime_characteristics) || (BOTTOM && BOTTOM.regime_characteristics);
    if (!rc) {{ document.getElementById('regime-means-2d-section').style.display = 'none'; return; }}
    const hasPCA = Object.values(rc).some(v => v && v.pca2d_x);
    if (!hasPCA) {{ document.getElementById('regime-means-2d-section').style.display = 'none'; return; }}

    document.getElementById('regime-means-2d-section').style.display = '';

    const rColors = [
        '#C4736A','#8FAE8B','#7B9CB5','#C4A456','#9B7BB5',
        '#B5877B','#5C8C84','#C4956A','#7B8CB5','#A57BAA',
        '#7BA57B','#B5A47B','#8C5C5C','#5C7B8C','#AAA57B','#7B9BAA',
    ];

    // Generic scatter plot builder
    function plotScatter(divId, meanX, meanY, sampleX, sampleY, sampleRegime, stds, d, tag, axisLabels) {{
        const K = meanX.length;
        const traces = [];

        if (sampleX && sampleX.length > 0 && sampleRegime) {{
            const regimeGroups = {{}};
            for (let i = 0; i < sampleX.length; i++) {{
                const r = sampleRegime[i];
                if (!regimeGroups[r]) regimeGroups[r] = {{ x: [], y: [] }};
                regimeGroups[r].x.push(sampleX[i]);
                regimeGroups[r].y.push(sampleY[i]);
            }}
            Object.keys(regimeGroups).sort((a, b) => +a - +b).forEach(rStr => {{
                const r = +rStr;
                const pts = regimeGroups[r];
                traces.push({{
                    x: pts.x, y: pts.y, mode: 'markers', type: 'scatter',
                    name: `R${{r}}`,
                    marker: {{ color: rColors[r % rColors.length], size: 4, opacity: 0.30 }},
                    hovertemplate: `Regime ${{r}}<extra></extra>`, showlegend: true,
                }});
            }});
        }}

        // Regime mean diamonds
        traces.push({{
            x: meanX, y: meanY, mode: 'markers+text', type: 'scatter',
            name: 'means',
            text: meanX.map((_, k) => String(k)),
            textposition: 'top center',
            textfont: {{ size: 9, color: '#2C2C2E' }},
            marker: {{
                color: meanX.map((_, k) => rColors[k % rColors.length]),
                size: sampleX && sampleX.length > 0 ? 16 : stds.map(s => {{
                    const minS = Math.min(...stds, 0), rng = Math.max(Math.max(...stds) - minS, 1e-10);
                    return 10 + 18 * (s - minS) / rng;
                }}),
                symbol: 'diamond', opacity: 1.0, line: {{ color: '#fff', width: 2 }},
            }},
            customdata: meanX.map((_, k) => [k,
                stds[k] ? stds[k].toFixed(4) : '?',
                d.regime_avg_variance ? d.regime_avg_variance[k].toFixed(5) : '?']),
            hovertemplate: 'Regime %{{customdata[0]}}<br>std_2d=%{{customdata[1]}}<br>avg_var=%{{customdata[2]}}<extra></extra>',
            showlegend: false,
        }});

        const sepRatio = d.separation_ratio != null ? d.separation_ratio : '?';
        Plotly.newPlot(divId, traces, {{
            ...plotLayout,
            title: {{ text: `sep_ratio=${{sepRatio}}`, font: {{ size: 10 }}, x: 0.5 }},
            xaxis: {{ ...plotLayout.xaxis, title: axisLabels[0], type: 'linear' }},
            yaxis: {{ ...plotLayout.yaxis, title: axisLabels[1], type: 'linear' }},
            margin: {{ t: 30, b: 50, l: 55, r: 20 }},
            legend: {{ font: {{ size: 8 }}, x: 1.0, xanchor: 'right' }},
        }}, {{ responsive: true }});
    }}

    ['FA','SR','AC','UV'].forEach(tag => {{
        const cIdx = TAGS.indexOf(tag);
        if (cIdx < 0) return;
        const d = rc[String(cIdx)];
        if (!d || !d.pca2d_x) return;

        const K = d.pca2d_x.length;
        const ev = d.pca2d_ev || [0, 0];
        const pct0 = ev[0] != null ? (ev[0] * 100).toFixed(1) : '?';
        const pct1 = ev[1] != null ? (ev[1] * 100).toFixed(1) : '?';

        // PCA plot (always)
        plotScatter(`regime-pca-${{tag}}`,
            d.pca2d_x, d.pca2d_y,
            d.pca_sample_x || null, d.pca_sample_y || null, d.pca_sample_regime || null,
            d.pca2d_regime_std || new Array(K).fill(0),
            d, tag,
            [`PC0 (${{pct0}}%)`, `PC1 (${{pct1}}%)`]);

    }});
}}

// ══════════════════════════════════════════════════════════════════
//  SOFT ACTIVATION PROFILES: layer×regime heatmap correct vs incorrect
// ══════════════════════════════════════════════════════════════════
function renderSoftProfiles() {{
    const mdBot = getBottomMd();
    const sp = (mdBot && mdBot.soft_profiles) || (BOTTOM && BOTTOM.soft_profiles);
    if (!sp) {{ document.getElementById('soft-profiles-section').style.display = 'none'; return; }}

    document.getElementById('soft-profiles-section').style.display = '';
    const grid = document.getElementById('soft-profiles-grid');

    const greenSeq = [[0,'#F5F0EB'],[0.33,'#D0D8CC'],[0.67,'#96B09E'],[1,'#5A7A6A']];
    const roseSeq  = [[0,'#F5F0EB'],[0.33,'#E0CACB'],[0.67,'#C89090'],[1,'#9E6870']];

    // Compute global max absolute diff across all 4 categories for shared color scale
    let globalMaxAbsDiff = 0;
    ['FA','SR','AC','UV'].forEach(tag => {{
        const d = sp[tag];
        if (!d || !d.diff_correct_minus_incorrect) return;
        d.diff_correct_minus_incorrect.forEach(row => row.forEach(v => {{
            const abs = Math.abs(v);
            if (abs > globalMaxAbsDiff) globalMaxAbsDiff = abs;
        }}));
    }});
    if (globalMaxAbsDiff < 0.01) globalMaxAbsDiff = 0.1;  // fallback

    ['FA','SR','AC','UV'].forEach(tag => {{
        const d = sp[tag];
        if (!d || !d.correct || !d.incorrect) return;

        const profC = d.correct.mean_profile;
        const profI = d.incorrect.mean_profile;
        const diff  = d.diff_correct_minus_incorrect;
        const topDisc = d.top_discriminative || [];
        if (!profC || !profI || !diff) return;

        const L = profC.length;
        const K = profC[0].length;
        const xTicks = Array.from({{length: K}}, (_, i) => 'R' + i);
        const yTicks = Array.from({{length: L}}, (_, i) => i + 1);

        // Top discriminative summary
        let discStr = '';
        if (topDisc.length > 0) {{
            discStr = topDisc.slice(0, 3).map(dd =>
                `L${{dd.layer+1}}\u00b7R${{dd.regime}} (${{dd.diff > 0 ? '+' : ''}}${{dd.diff.toFixed(3)}})`
            ).join(', ');
            discStr = `<span style="font-size:11px; color:#AEAEB2;"> &nbsp;Top: ${{discStr}}</span>`;
        }}

        const nc = d.correct.n_steps;
        const ni = d.incorrect.n_steps;

        const rowDiv = document.createElement('div');
        rowDiv.style.cssText = 'margin-bottom:24px;';
        rowDiv.innerHTML = `
            <h5 style="font-size:13px; color:#6B6B70; margin-bottom:6px;">${{tag}} &mdash; ${{NAMES[tag] || tag}}${{discStr}}</h5>
            <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <div style="flex:1; min-width:120px;">
                    <div style="font-size:11px; font-weight:600; color:${{mGreen}}; text-align:center; margin-bottom:2px;">Correct (n=${{nc}})</div>
                    <div id="sp-${{tag}}-corr" style="height:420px;"></div>
                </div>
                <div style="flex:1; min-width:120px;">
                    <div style="font-size:11px; font-weight:600; color:${{mRose}}; text-align:center; margin-bottom:2px;">Incorrect (n=${{ni}})</div>
                    <div id="sp-${{tag}}-incorr" style="height:420px;"></div>
                </div>
                <div style="flex:1; min-width:120px;">
                    <div style="font-size:11px; font-weight:600; color:#6B6B70; text-align:center; margin-bottom:2px;">Correct &minus; Incorrect</div>
                    <div id="sp-${{tag}}-diff" style="height:420px;"></div>
                </div>
            </div>
        `;
        grid.appendChild(rowDiv);

        const hmLayout = {{
            ...plotLayout,
            margin: {{ t: 6, b: 40, l: 50, r: 10 }},
            xaxis: {{ ...plotLayout.xaxis, title: {{ text: 'Regime', font: {{ size: 10 }} }}, tickfont: {{ size: 9 }} }},
            yaxis: {{ ...plotLayout.yaxis, title: {{ text: 'Layer', font: {{ size: 10 }} }}, autorange: 'reversed', tickfont: {{ size: 8 }},
                     dtick: L > 40 ? 10 : 5 }},
        }};

        const makeHM = (mat, cs, divId, zmin, zmax, stdMat) => {{
            const trace = {{
                z: mat, x: xTicks, y: yTicks,
                type: 'heatmap', colorscale: cs,
                zmin, zmax,
                showscale: true,
                colorbar: {{ thickness: 10, len: 0.5 }},
            }};
            if (stdMat) {{
                trace.customdata = mat.map((row, i) => row.map((_, j) => stdMat[i] ? stdMat[i][j] : 0));
                trace.hovertemplate = 'Layer %{{y}}, %{{x}}: %{{z:.4f}} ±%{{customdata:.4f}}<extra></extra>';
            }} else {{
                trace.hovertemplate = 'Layer %{{y}}, %{{x}}: %{{z:.4f}}<extra></extra>';
            }}
            Plotly.newPlot(divId, [trace], hmLayout, {{ responsive: true }});
        }};

        const stdC = d.correct && d.correct.std_profile;
        const stdI = d.incorrect && d.incorrect.std_profile;
        makeHM(profC, greenSeq,    `sp-${{tag}}-corr`,   0,    1, stdC || null);
        makeHM(profI, roseSeq,     `sp-${{tag}}-incorr`, 0,    1, stdI || null);

        // Diff heatmap with per-layer KL annotation on peak layer
        const eps = 1e-9;
        let maxKL = -1, maxKLLayer = 0;
        for (let l = 0; l < L; l++) {{
            let kl = 0;
            for (let k = 0; k < K; k++) {{
                const p = profC[l][k] + eps;
                const q = profI[l][k] + eps;
                kl += p * Math.log(p / q);
            }}
            if (kl > maxKL) {{ maxKL = kl; maxKLLayer = l; }}
        }}
        const diffLayout = {{
            ...hmLayout,
            annotations: [{{
                x: K - 0.5, y: maxKLLayer,
                xref: 'x', yref: 'y', xanchor: 'right',
                text: `\u25C0 L${{maxKLLayer}} KL=${{maxKL.toFixed(3)}}`,
                showarrow: false,
                font: {{ size: 10, color: '#C44E52', family: 'monospace' }},
                bgcolor: 'rgba(255,255,255,0.8)',
            }}],
        }};
        const stdDiff = d.std_diff || null;
        const diffTrace = {{
            z: diff, x: xTicks, y: yTicks,
            type: 'heatmap', colorscale: morandiDivR,
            zmin: -globalMaxAbsDiff, zmax: globalMaxAbsDiff,
            showscale: true,
            colorbar: {{ thickness: 10, len: 0.5 }},
        }};
        if (stdDiff) {{
            diffTrace.customdata = diff.map((row, i) => row.map((_, j) => stdDiff[i] ? stdDiff[i][j] : 0));
            diffTrace.hovertemplate = 'Layer %{{y}}, %{{x}}: %{{z:.4f}} ±%{{customdata:.4f}}<extra></extra>';
        }} else {{
            diffTrace.hovertemplate = 'Layer %{{y}}, %{{x}}: %{{z:.4f}}<extra></extra>';
        }}
        Plotly.newPlot(`sp-${{tag}}-diff`, [diffTrace], diffLayout, {{ responsive: true }});
    }});
}}

// ══════════════════════════════════════════════════════════════════
//  TEMPORAL STRUCTURE: Binning, Transitions, Dwell, First-Occurrence
// ══════════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════════
//  STEP TRAJECTORY: layer-by-layer hidden state movement
// ══════════════════════════════════════════════════════════════════

// 16-color qualitative palette for up to 16 regimes
const REGIME_PAL = [
    '#4C72B0','#DD8452','#55A868','#C44E52','#8172B3',
    '#937860','#DA8BC3','#8C8C8C','#CCB974','#64B5CD',
    '#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B',
];

function renderStepTrajectory() {{
    const mdBot = getBottomMd();
    const st = (mdBot && mdBot.step_trajectories) || (BOTTOM && BOTTOM.step_trajectories);
    if (!st || Object.keys(st).length === 0) {{ document.getElementById('step-trajectory-section').style.display = 'none'; return; }}
    document.getElementById('step-trajectory-section').style.display = '';

    const CAT_COL = {{FA:'#C48B9F', SR:'#8B9DAA', AC:'#8BAA92', UV:'#C9A84C'}};
    let currentTag = Object.keys(st)[0];
    let highlightIdx = -1;

    // ── Category tab buttons ──────────────────────────────────────────────
    const tabsEl = document.getElementById('traj-cat-tabs');
    Object.keys(st).forEach(tag => {{
        const btn = document.createElement('button');
        const col = CAT_COL[tag] || '#7C8B9A';
        btn.dataset.tag = tag;
        btn.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;
            background:${{col}};margin-right:5px;vertical-align:middle;opacity:0.8;"></span>${{tag}}`;
        const isFirst = tag === currentTag;
        btn.style.cssText = `font-size:12px;padding:5px 16px;border-radius:20px;cursor:pointer;
            transition:all 0.15s;border:1.5px solid ${{isFirst ? '#7C8B9A' : '#D1CCC6'}};
            background:${{isFirst ? '#E8E4E0' : '#FAF7F4'}};color:#3C3C3E;
            font-weight:${{isFirst ? '600' : '400'}};`;
        btn.onclick = () => {{
            currentTag = tag; highlightIdx = -1;
            [...tabsEl.children].forEach(b => {{
                const a = b.dataset.tag === tag;
                b.style.background = a ? '#E8E4E0' : '#FAF7F4';
                b.style.fontWeight = a ? '600' : '400';
                b.style.borderColor = a ? '#7C8B9A' : '#D1CCC6';
            }});
            renderForTag(tag);
        }};
        tabsEl.appendChild(btn);
    }});

    function renderForTag(tag) {{
        const data = st[tag];
        if (!data) return;
        const samples = data.samples;
        const K = data.regime_means_x.length;

        // ── Build chips ───────────────────────────────────────────────────
        const correctEl   = document.getElementById('traj-correct-list');
        const incorrectEl = document.getElementById('traj-incorrect-list');
        correctEl.innerHTML = ''; incorrectEl.innerHTML = '';

        function makeLabel(text, color) {{
            const d = document.createElement('div');
            d.className = 'traj-group-label';
            d.style.color = color; d.textContent = text; return d;
        }}
        let hasCorr = false, hasIncorr = false;
        samples.forEach((s, i) => {{
            const isCorr = s.is_correct === true;
            const isIncorr = s.is_correct === false;
            if (!isCorr && !isIncorr) return;
            const color = isCorr ? '#5B8DB8' : '#D4885C';
            const sym = isCorr ? '\u2713' : '\u2717';
            const chip = document.createElement('div');
            chip.className = 'traj-chip'; chip.dataset.idx = i;
            chip.innerHTML = `<span class="traj-chip-dot" style="background:${{color}};"></span>
                <span style="color:#3C3C3E;line-height:1.35;">
                    <b>${{sym}} step ${{s.step_idx}}</b>
                    <span style="color:#8E8E93;font-size:10px;"> · ${{s.n_transitions}}T · ${{s.n_layers}}L</span>
                </span>`;
            chip.title = s.regime_summary || '';
            chip.onclick = () => {{
                const idx = parseInt(chip.dataset.idx);
                const wasActive = highlightIdx === idx;
                highlightIdx = wasActive ? -1 : idx;
                document.querySelectorAll('.traj-chip').forEach(c => c.classList.remove('active'));
                if (!wasActive) chip.classList.add('active');
                updateOpacities();
            }};
            if (isCorr) {{
                if (!hasCorr) {{ correctEl.appendChild(makeLabel('\u2713  CORRECT', '#3A6B96')); hasCorr = true; }}
                correctEl.appendChild(chip);
            }} else {{
                if (!hasIncorr) {{ incorrectEl.appendChild(makeLabel('\u2717  INCORRECT', '#A06030')); hasIncorr = true; }}
                incorrectEl.appendChild(chip);
            }}
        }});

        // ── Build Plotly traces ───────────────────────────────────────────
        const traces = [];

        // Trace 0: regime centroids (always visible)
        traces.push({{
            x: data.regime_means_x, y: data.regime_means_y,
            mode: 'markers+text', type: 'scatter', name: 'Centroids',
            text: Array.from({{length: K}}, (_, k) => `R${{k}}`),
            textposition: 'top center',
            textfont: {{ size: 10, color: '#888' }},
            marker: {{
                symbol: 'star', size: 22, opacity: 1,
                color: Array.from({{length: K}}, (_, k) => REGIME_PAL[k % REGIME_PAL.length]),
                line: {{ color: '#666', width: 1.5 }},
            }},
            hovertemplate: 'Centroid R%{{pointNumber}}<extra></extra>',
            showlegend: false,
        }});

        const sampleOffset = 1;  // trace 0 = centroids

        // Pick up to 5 correct + 5 incorrect as default-visible samples
        const corrIdx = [], incorrIdx = [];
        samples.forEach((s, i) => {{
            if (s.is_correct === true) corrIdx.push(i);
            else if (s.is_correct === false) incorrIdx.push(i);
        }});
        // Shuffle deterministically and pick 5 each
        function shuffle(arr) {{ for (let i = arr.length - 1; i > 0; i--) {{ const j = (i * 7 + 3) % (i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; }} return arr; }}
        const defaultVisible = new Set([...shuffle(corrIdx).slice(0, 2), ...shuffle(incorrIdx).slice(0, 2)]);

        // Individual sample traces (3 per sample: line, markers, labels)
        // Default: show random 5 correct + 5 incorrect. Click chip to highlight one.
        samples.forEach((sample, i) => {{
            const isCorr = sample.is_correct === true;
            const lineCol = isCorr ? '#5B8DB8' : '#D4885C';
            const L = sample.n_layers;
            const xs = sample.x, ys = sample.y, regs = sample.regimes;

            const defVis = defaultVisible.has(i);
            const defOp = defVis ? 0.75 : 0;

            // Path line
            traces.push({{
                x: xs, y: ys, mode: 'lines', type: 'scatter',
                line: {{ color: lineCol, width: 1.5, dash: isCorr ? 'solid' : 'dot' }},
                opacity: defOp, hoverinfo: 'skip', showlegend: false,
            }});

            // Markers colored by regime
            traces.push({{
                x: xs, y: ys, mode: 'markers', type: 'scatter',
                marker: {{
                    color: regs.map(r => REGIME_PAL[r % REGIME_PAL.length]),
                    size:   xs.map((_, l) => l === 0 ? 11 : (l > 0 && regs[l] !== regs[l-1] ? 14 : 7)),
                    symbol: xs.map((_, l) => l === 0 ? 'circle-open'
                                           : (l > 0 && regs[l] !== regs[l-1] ? 'diamond' : 'circle')),
                    line: {{
                        color: xs.map((_, l) => (l > 0 && regs[l] !== regs[l-1]) ? '#222' : 'rgba(0,0,0,0)'),
                        width: xs.map((_, l) => (l > 0 && regs[l] !== regs[l-1]) ? 1.5 : 0),
                    }},
                }},
                opacity: defOp,
                text: xs.map((_, l) => `${{sample.label}}<br>Layer ${{l+1}} → R${{regs[l]}}`
                    + (l > 0 && regs[l] !== regs[l-1] ? '  ◄ transition' : '')
                    + (l === 0 ? '  <i>(start)</i>' : l === L-1 ? '  <i>(end)</i>' : '')),
                hovertemplate: '%{{text}}<extra></extra>',
                showlegend: false,
            }});

            // Regime labels at every layer
            traces.push({{
                x: xs, y: ys, mode: 'text', type: 'scatter',
                text: regs.map(r => 'R' + r),
                textposition: 'top right',
                textfont: {{ size: 7, color: regs.map(r => REGIME_PAL[r % REGIME_PAL.length]), family: 'monospace' }},
                opacity: defOp > 0 ? 0.65 : 0, hoverinfo: 'skip', showlegend: false,
            }});
        }});

        Plotly.newPlot('traj-plot', traces, {{
            ...plotLayout,
            title: false,
            xaxis: {{ ...plotLayout.xaxis, title: 'PCA dim 0', zeroline: false, type: 'linear' }},
            yaxis: {{ ...plotLayout.yaxis, title: 'PCA dim 1', zeroline: false, type: 'linear' }},
            margin: {{ ...plotLayout.margin, t: 20, b: 50 }},
            showlegend: false,
        }}, {{ responsive: true }});
    }}

    function updateOpacities() {{
        const data = st[currentTag];
        if (!data) return;
        const N = data.samples.length;
        const off = sampleOffset;
        const indices = [], opacities = [];
        for (let i = 0; i < N; i++) {{
            if (highlightIdx === -1) {{
                // No selection: show default visible set
                const vis = defaultVisible.has(i);
                indices.push(off + 3 * i, off + 3 * i + 1, off + 3 * i + 2);
                opacities.push(vis ? 0.75 : 0, vis ? 0.75 : 0, vis ? 0.65 : 0);
            }} else {{
                const isHighlighted = (i === highlightIdx);
                indices.push(off + 3 * i, off + 3 * i + 1, off + 3 * i + 2);
                opacities.push(isHighlighted ? 0.95 : 0, isHighlighted ? 0.95 : 0, isHighlighted ? 0.85 : 0);
            }}
        }}
        Plotly.restyle('traj-plot', {{ opacity: opacities }}, indices);
    }}

    document.getElementById('traj-show-all-btn').onclick = () => {{
        highlightIdx = -1;
        document.querySelectorAll('.traj-chip').forEach(c => c.classList.remove('active'));
        updateOpacities();
    }};

    renderForTag(currentTag);
}}

// ══════════════════════════════════════════════════════════════════
//  COMBINED TRAJECTORY: all categories in one plot
// ══════════════════════════════════════════════════════════════════


// ══════════════════════════════════════════════════════════════════
const TAG_ID = {{}};
TAGS.forEach((t, i) => TAG_ID[t] = i);

function idToTag(id) {{ return TAGS[id] || '?'; }}

// ══════════════════════════════════════════════════════════════════
//  INDIVIDUAL: Step detail panel (click-to-open bottom analysis)
// ══════════════════════════════════════════════════════════════════

function getBottomMetrics() {{
    const mdBot = getBottomMd();
    return (mdBot && mdBot.metrics) || (BOTTOM && BOTTOM.metrics) || {{}};
}}

// Look up per-step hidden state data from BOTTOM.sampled_step_details
// BOTTOM now merges step details from ALL runs, so exact label matching should work.
function findStepDetail(labels, idx) {{
    const rsd = BOTTOM && BOTTOM.sampled_step_details;
    if (!rsd) return null;

    const _extract = (seq) => {{
        if (!seq || !seq.steps) return null;
        const step = seq.steps.find(s => s.step === idx);
        return step ? {{ step, allSteps: seq.steps, seqInfo: seq }} : null;
    }};

    // Strategy 1: Exact label match (should work now that BOTTOM has all runs)
    for (const group of ['correct', 'long_fail', 'short_fail']) {{
        const seqs = rsd[group] || [];
        for (const seq of seqs) {{
            if (seq.labels && seq.labels.length === labels.length &&
                seq.labels.every((l, i) => l === labels[i])) {{
                return _extract(seq);
            }}
        }}
    }}

    // Strategy 2: Best overlap fallback (defensive)
    let bestSeq = null, bestOverlap = 0;
    for (const group of ['correct', 'long_fail', 'short_fail']) {{
        const seqs = rsd[group] || [];
        for (const seq of seqs) {{
            if (!seq.labels || !seq.steps) continue;
            const n = Math.min(seq.labels.length, labels.length);
            let overlap = 0;
            for (let i = 0; i < n; i++) {{ if (seq.labels[i] === labels[i]) overlap++; }}
            if (overlap > bestOverlap) {{ bestOverlap = overlap; bestSeq = seq; }}
        }}
    }}
    if (bestSeq && bestOverlap >= Math.min(labels.length, 3)) {{
        return _extract(bestSeq);
    }}

    return null;
}}

function renderStepDetail(labels, idx, panelId) {{
    /* Build and show the detail panel for a clicked step. */
    const panel = document.getElementById(panelId);
    if (!panel) return;

    const cat = idToTag(labels[idx]);
    const catFull = NAMES[cat] || cat;
    const color = COLORS[cat] || '#666';
    const bm = getBottomMetrics();
    const hsData = findStepDetail(labels, idx);

    let html = '<div class="step-detail-header">';
    html += `<h4><span style="color:${{color}}">■</span> Step ${{idx+1}}: ${{catFull}} (${{cat}})</h4>`;
    html += `<span class="step-detail-close" onclick="closeStepDetail('${{panelId}}')">✕</span>`;
    html += '</div>';

    // ═══════════════════════════════════════════════════════════════
    //  Per-step hidden state analysis (actual data from this sequence)
    // ═══════════════════════════════════════════════════════════════
    if (hsData && hsData.step) {{
        const s = hsData.step;

        html += '<div class="sd-card" style="margin-bottom:14px; border-color:' + mSteel + ';">';
        html += '<h5 style="color:' + mSteel + ';">Hidden State Analysis (this step)</h5>';

        // Regime info
        html += `<div class="sd-metric"><span class="sd-label">Dominant regime (across layers)</span>`;
        html += `<span class="sd-value">Regime #${{s.regime_dominant}}</span></div>`;

        // Regime path z-score vs correct
        if (s.regime_path_zscore != null) {{
            const rpColor = s.regime_path_zscore < -2.0 ? mRose : (s.regime_path_zscore < -1.0 ? mGold : mGreen);
            html += `<div class="sd-metric"><span class="sd-label">Regime path (vs correct ${{cat}})</span>`;
            html += `<span class="sd-value" style="color:${{rpColor}}">${{s.regime_path_zscore >= 0 ? '+' : ''}}${{s.regime_path_zscore.toFixed(2)}}σ</span></div>`;
        }}

        html += '</div>';

        // ── Per-layer regime visualization ──
        if (s.layer_data) {{
            const ld = s.layer_data;
            const nLayers = ld.regimes_per_layer.length;
            const regimeColors = ['#8BAAB5', '#8BAA92', '#C4B176', '#C48B9F', '#8B7DAA', '#7CA8A0', '#B5A89A', '#A2A8B0'];
            html += '<div class="sd-card" style="margin-bottom:14px; border-color:' + mPlum + ';">';
            html += `<h5 style="color:${{mPlum}};">Per-Layer Analysis (${{nLayers}} transformer layers)</h5>`;

            // Layer regime strip
            html += '<div style="font-size:11px; color:#8E8E93; margin-bottom:4px;">Regime per layer:</div>';
            html += '<div class="layer-strip">';
            for (let l = 0; l < nLayers; l++) {{
                const r = ld.regimes_per_layer[l];
                const bg = regimeColors[r % regimeColors.length];
                const tip = `Layer ${{l+1}}: regime ${{r}}`;
                html += `<div class="layer-cell" style="background:${{bg}}">${{r}}<div class="layer-tip">${{tip}}</div></div>`;
            }}
            html += '</div>';

            html += '</div>';
        }}

    }}

    panel.innerHTML = html;
    panel.classList.add('open');

    panel.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
}}

function closeStepDetail(panelId) {{
    const panel = document.getElementById(panelId);
    if (panel) {{
        panel.classList.remove('open');
        panel.innerHTML = '';
    }}
    // Deselect all steps in the parent timeline
    const parent = panel.parentElement;
    if (parent) parent.querySelectorAll('.timeline-step.selected').forEach(s => s.classList.remove('selected'));
}}

// ══════════════════════════════════════════════════════════════════
//  INDIVIDUAL: Timeline rendering
// ══════════════════════════════════════════════════════════════════

// Global counter to give each timeline a unique ID
let _tlIdCounter = 0;

function renderTimeline(labels, containerId) {{
    const el = document.getElementById(containerId);
    if (!el) return;

    const tlId = 'tl-' + (_tlIdCounter++);
    const detailId = tlId + '-detail';

    let html = '<div class="timeline" id="' + tlId + '">';

    for (let i = 0; i < labels.length; i++) {{
        const cat = idToTag(labels[i]);
        const color = COLORS[cat] || '#666';
        const textColor = '#fff';

        // Build tooltip (brief — details on click)
        let tooltipHtml = `<strong>Step ${{i+1}}</strong>: ${{NAMES[cat] || cat}}`;
        if (i === 0) tooltipHtml += '<br><em>Start</em>';
        tooltipHtml += `<br><em style="color:${{mSteel}}">click for details</em>`;

        html += `<div class="timeline-step" style="background:${{color}};color:${{textColor}}"
                      data-idx="${{i}}" data-tlid="${{tlId}}" data-detailid="${{detailId}}">
            ${{cat}}<div class="tooltip">${{tooltipHtml}}</div></div>`;
    }}
    html += '</div>';

    // Step detail panel placeholder
    html += `<div class="step-detail" id="${{detailId}}"></div>`;

    el.innerHTML = html;

    // Store labels on the timeline element for click handler
    const tlEl = document.getElementById(tlId);
    tlEl._labels = labels;

    // Attach click handlers
    document.getElementById(tlId).querySelectorAll('.timeline-step').forEach(stepEl => {{
        stepEl.addEventListener('click', function(e) {{
            e.stopPropagation();
            const idx = parseInt(this.dataset.idx);
            const tlId = this.dataset.tlid;
            const detailId = this.dataset.detailid;
            const tl = document.getElementById(tlId);

            // Toggle selection
            const wasSelected = this.classList.contains('selected');
            tl.querySelectorAll('.timeline-step.selected').forEach(s => s.classList.remove('selected'));

            if (wasSelected) {{
                closeStepDetail(detailId);
            }} else {{
                this.classList.add('selected');
                renderStepDetail(tl._labels, idx, detailId);
            }}
        }});
    }});
}}

// (renderWarningReport removed)

// (renderWarningValidation removed)

// (renderValidationSummary removed)

function renderExamples() {{
    const container = document.getElementById('examples-container');
    const mdTop = getTopMd();
    const reps = (mdTop && mdTop.sampled_sequences) || TOP.sampled_sequences || {{}};

    // Collect all sequences from all groups, tagged with their group
    const allSeqs = [];
    const groups = ['correct', 'long_fail', 'short_fail'];
    groups.forEach(g => {{
        const seqs = reps[g] || [];
        seqs.forEach(seq => allSeqs.push({{ ...seq, _group: g }}));
    }});

    // Limit to 10 total
    const shown = allSeqs.slice(0, 10);

    if (shown.length === 0) {{
        container.innerHTML = '<p style="color:#AEAEB2;">No representative sequences available.</p>';
        return;
    }}

    container.innerHTML = shown.map((seq, idx) => {{
        const nSteps = seq.labels.length;

        // Step-by-step category strip: one thin colored cell per step
        let stepStrip = '<div style="display:flex; height:8px; border-radius:3px; overflow:hidden; margin:4px 0;" title="Category per step">';
        seq.labels.forEach((l, i) => {{
            const t = idToTag(l);
            stepStrip += `<div style="flex:1; background:${{COLORS[t] || '#888'}};" title="Step ${{i+1}}: ${{t}}"></div>`;
        }});
        stepStrip += '</div>';

        // Category composition text
        const catCounts = {{}};
        TAGS.forEach(t => catCounts[t] = 0);
        seq.labels.forEach(l => {{ const t = idToTag(l); if (catCounts[t] !== undefined) catCounts[t]++; }});
        let compText = TAGS.filter(t => catCounts[t] > 0)
            .map(t => `<span style="color:${{COLORS[t]}};font-weight:600;">${{t}}</span> ${{(catCounts[t]/nSteps*100).toFixed(0)}}%`)
            .join(' &nbsp; ');

        // Top transitions
        const transCounts = {{}};
        for (let i = 1; i < nSteps; i++) {{
            const from = idToTag(seq.labels[i-1]);
            const to = idToTag(seq.labels[i]);
            const key = from + '\u2192' + to;
            transCounts[key] = (transCounts[key] || 0) + 1;
        }}
        const topTrans = Object.entries(transCounts).sort((a,b) => b[1] - a[1]).slice(0, 3);
        let transText = topTrans.map(([k,c]) => `${{k}} ${{c}}\u00d7`).join(', ');

        return `
        <div class="example-card">
            <div style="cursor:pointer;" onclick="this.parentElement.classList.toggle('expanded')">
                <div class="example-header">
                    <div>
                        <span style="color:#8E8E93; font-size:12px;">
                            Length: ${{seq.path_length || nSteps}}${{seq.sample_idx != null ? ' &nbsp;#' + seq.sample_idx : ''}}
                        </span>
                    </div>
                    <span style="color:#AEAEB2; font-size:18px;">\u25BE</span>
                </div>
                <div class="example-question">${{escapeHtml(seq.question || '(no question text)')}}</div>
                ${{stepStrip}}
            </div>
            <div class="detail-panel">
                <div style="font-size:11px; color:#6B6B70; margin-bottom:2px;">${{compText}}</div>
                <div style="font-size:11px; color:#8E8E93; margin-bottom:8px;">Transitions: ${{transText || '—'}}</div>
                <div id="timeline-all-${{idx}}"></div>
            </div>
        </div>
    `;
    }}).join('');

    // Render timelines
    shown.forEach((seq, idx) => {{
        renderTimeline(seq.labels, `timeline-all-${{idx}}`);
    }});
}}

function escapeHtml(str) {{
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}}

// ══════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {{
    {"" if is_fixed else "buildFilterBar();"}
    renderAll();
}});
</script>
</body>
</html>"""

    return html


def _build_per_md_data(top_data, bottom_data, model, dataset):
    """Build TOP and BOTTOM objects containing only one model/dataset combo."""
    md_key = f"{model}/{dataset}"

    # TOP: per_model_dataset entry + aggregate fallbacks only when per-md lacks them
    top_md = (top_data.get("per_model_dataset") or {}).get(md_key, {})

    # Always copy small / structural fields
    top_slim = {}
    for k in ("config", "models", "datasets"):
        if k in top_data:
            top_slim[k] = top_data[k]
    # Fields with per-md fallback: only include aggregate when per-md lacks them
    for k in ("1st_order_matrices", "2nd_order_matrices",
              "3rd_order_matrices", "4th_order_matrices", "5th_order_matrices",
              "start_probs", "population_stats", "group_sequence_counts",
              "stationary_distribution", "hitting_times_to_FA",
              "sampled_sequences"):
        if k not in top_md and k in top_data:
            top_slim[k] = top_data[k]

    top_slim["per_model_dataset"] = {md_key: top_md}
    top_slim["models"] = [model]
    top_slim["datasets"] = [dataset]

    # BOTTOM: keep aggregate metrics/warnings + just this combo's data
    bot_slim = {}
    if bottom_data:
        bot_pmd = (bottom_data.get("per_model_dataset") or {}).get(md_key, {})

        # Always copy small aggregate-only fields
        for k in ("n_runs", "models", "datasets",
                  "cross_seed_consistency"):
            if k in bottom_data:
                bot_slim[k] = bottom_data[k]

        # Fields used with fallback pattern (mdBot.X || BOTTOM.X):
        # only copy aggregate version when per_model_dataset entry lacks them.
        for k in ("metrics", "direction_correctness_comparison",
                  "category_distributions",
                  "regime_characteristics",
                  "regime_source_run", "cat_dist_source_run",
                  "regime_top_transitions", "regime_top_source_run",
                  "step_trajectories", "step_trajectories_source_run",
                  "soft_profiles", "soft_profiles_source_run"):
            if k not in bot_pmd and k in bottom_data:
                bot_slim[k] = bottom_data[k]

        bot_slim["per_model_dataset"] = {md_key: bot_pmd}

        # sampled_step_details: prefer per-md, fall back to global
        rsd = bot_pmd.get("sampled_step_details")
        if not rsd:
            rsd = bottom_data.get("sampled_step_details")
        if rsd:
            bot_slim["sampled_step_details"] = rsd

        # per_seed_data: filter to this model/dataset only
        psd = bottom_data.get("per_seed_data", [])
        if psd:
            bot_slim["per_seed_data"] = [
                s for s in psd
                if s.get("model") == model and s.get("dataset") == dataset
            ]

    return top_slim, bot_slim


def _generate_index_html(models, datasets, top_data):
    """Generate index.html with PRISM logo, abstract, and model→dataset selector."""

    MODEL_FULL_NAMES = {
        "stratos": "Bespoke-Stratos-7B",
        "qwen": "Qwen3-1.7B",
        "openthinker": "OpenThinker-7B",
        "nemotron": "Llama-3.1-Nemotron-Nano-4B-v1.1",
    }
    DATASET_FULL_NAMES = {
        "math500": "MATH500",
        "gpqa_diamond": "GPQA-Diamond",
        "aime24": "AIME24",
        "tiger": "WebInstruct-verified",
    }

    # Fixed ordering: stratos first, gpqa first
    MODEL_ORDER = ["stratos", "qwen", "openthinker", "nemotron"]
    DATASET_ORDER = ["gpqa_diamond", "math500", "aime24", "tiger"]

    model_datasets = {}
    for model in MODEL_ORDER:
        if model not in models:
            continue
        ds_list = []
        for dataset in DATASET_ORDER:
            if dataset not in datasets:
                continue
            fname = f"{model}_{dataset}.html"
            ds_list.append((dataset, fname))
        if ds_list:
            model_datasets[model] = ds_list

    config = top_data.get("config", {})
    n_runs = config.get("num_runs", "?")
    n_total = sum(len(v) for v in model_datasets.values())

    # Build model buttons and dataset panels
    model_buttons = []
    dataset_panels = []
    for i, (model, ds_list) in enumerate(model_datasets.items()):
        full_name = MODEL_FULL_NAMES.get(model, model)
        active = "active" if i == 0 else ""
        model_buttons.append(
            f'<button class="model-btn {active}" onclick="selectModel(\'{model}\')" '
            f'id="btn-{model}">{full_name}</button>')

        display = "flex" if i == 0 else "none"
        ds_links = []
        for dataset, fname in ds_list:
            ds_full = DATASET_FULL_NAMES.get(dataset, dataset)
            ds_links.append(
                f'<a href="{fname}" class="ds-link">{ds_full}</a>')
        dataset_panels.append(
            f'<div class="ds-panel" id="panel-{model}" style="display:{display};">'
            f'{"".join(ds_links)}</div>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PRISM Analysis Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', system-ui, sans-serif;
        background: #F5F3F0; color: #2C2C2E; -webkit-font-smoothing: antialiased; }}

.hero {{ background: #fff; border-bottom: 1px solid #E8E4E0; padding: 48px 24px; text-align: center; }}
.hero h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 20px; }}
.hero img {{ max-width: 800px; width: 95%; margin-bottom: 28px; }}
.hero .abstract {{ max-width: 800px; margin: 0 auto; text-align: justify; font-size: 14px;
                   line-height: 1.7; color: #4A4A4C; }}

.container {{ max-width: 900px; margin: 32px auto; padding: 0 24px; }}

.section-title {{ font-size: 18px; font-weight: 600; margin-bottom: 16px; }}

.all-link {{ display: block; background: #fff; border: 1px solid #7C8B9A; border-radius: 12px;
             padding: 20px; text-decoration: none; color: inherit; margin-bottom: 24px;
             text-align: center; transition: all 0.2s ease; }}
.all-link:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.08); transform: translateY(-1px); }}
.all-link .label {{ font-size: 16px; font-weight: 600; }}
.all-link .sub {{ font-size: 13px; color: #8E8E93; margin-top: 4px; }}

.selector {{ background: #fff; border: 1px solid #E8E4E0; border-radius: 12px;
             padding: 24px; }}

.model-bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
.model-btn {{ padding: 10px 20px; border: 1px solid #E8E4E0; border-radius: 8px;
              background: #FAFAF8; cursor: pointer; font-size: 14px; font-weight: 500;
              color: #4A4A4C; transition: all 0.15s ease; }}
.model-btn:hover {{ border-color: #7C8B9A; }}
.model-btn.active {{ background: #2C2C2E; color: #fff; border-color: #2C2C2E; }}

.ds-panel {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.ds-link {{ display: block; padding: 14px 24px; background: #FAFAF8; border: 1px solid #E8E4E0;
            border-radius: 8px; text-decoration: none; color: #2C2C2E; font-size: 14px;
            font-weight: 500; transition: all 0.15s ease; }}
.ds-link:hover {{ border-color: #7C8B9A; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                  transform: translateY(-1px); }}
</style>
</head>
<body>

<div class="hero">
    <h1>PRISM Analysis Dashboard</h1>
    <img src="prism.png" alt="PRISM Framework" onerror="this.style.display='none'">
    <div class="abstract">
        Large language models increasingly solve complex problems by generating multi-step
        reasoning traces. Yet these traces are typically analyzed from only one of two perspectives:
        the sequence of tokens across different reasoning steps in the generated text, or the
        hidden-state vectors across model layers within one step. We introduce
        <strong>PRISM</strong> (<strong>P</strong>robabilistic <strong>R</strong>easoning
        <strong>I</strong>nspection through <strong>S</strong>emantic and
        <strong>I</strong>mplicit <strong>M</strong>odeling), a framework and diagnostic tool
        for jointly analyzing both levels, providing a unified view of how reasoning evolves
        across steps and layers.
    </div>
</div>

<div class="container">
    <a href="all.html" class="all-link">
        <div class="label">All Models &amp; Datasets (Aggregated)</div>
    </a>

    <div class="section-title">Select Model &amp; Dataset</div>
    <div class="selector">
        <div class="model-bar">
            {''.join(model_buttons)}
        </div>
        {''.join(dataset_panels)}
    </div>
</div>

<script>
function selectModel(model) {{
    document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + model).classList.add('active');
    document.querySelectorAll('.ds-panel').forEach(p => p.style.display = 'none');
    document.getElementById('panel-' + model).style.display = 'flex';
}}
</script>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML website for HHMM-GMM analysis")
    parser.add_argument("--top_json", type=str, required=True,
                        help="Path to aggregate_transitions.json")
    parser.add_argument("--bottom_json", type=str, default=None,
                        help="Path to aggregate_bottom.json (optional)")
    parser.add_argument("--output", type=str, default="index.html",
                        help="Output HTML file path (or directory for multi-page)")
    args = parser.parse_args()

    top_data = load_json(args.top_json)
    bottom_data = load_json(args.bottom_json)

    if not top_data:
        print(f"Error: could not load {args.top_json}")
        return

    # Discover model/dataset combinations
    top_pmd = top_data.get("per_model_dataset", {})
    bot_pmd = (bottom_data or {}).get("per_model_dataset", {})
    all_keys = sorted(set(list(top_pmd.keys()) + list(bot_pmd.keys())))

    if len(all_keys) <= 1:
        # Single combo or no per-md data: generate single file (legacy mode)
        html = generate_html(top_data, bottom_data)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        size_kb = os.path.getsize(args.output) / 1024
        print(f"[OK] Generated {args.output} ({size_kb:.0f} KB)")
        return

    # Multi-page mode: output directory
    out_dir = os.path.splitext(args.output)[0]
    if args.output.endswith(".html"):
        out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    models = sorted({k.split("/")[0] for k in all_keys})
    datasets = sorted({k.split("/", 1)[1] for k in all_keys})

    # Generate per-model/dataset pages
    for md_key in all_keys:
        model, dataset = md_key.split("/", 1)
        top_slim, bot_slim = _build_per_md_data(top_data, bottom_data, model, dataset)
        html = generate_html(top_slim, bot_slim,
                             fixed_model=model, fixed_dataset=dataset)
        fname = f"{model}_{dataset}.html"
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  [{model}/{dataset}] {fname} ({size_kb:.0f} KB)")

    # Generate "All" aggregate page — keep per_model_dataset for cross-MD std computation
    top_all = dict(top_data)
    top_all["models"] = sorted(models)
    top_all["datasets"] = sorted(datasets)
    bot_all = {}
    if bottom_data:
        # Exclude per_model_dataset, per_seed_data, and sampled_step_details
        # (large fields only needed in per-md pages) to keep aggregate page small.
        bot_all = {k: v for k, v in bottom_data.items()
                   if k not in ("per_model_dataset", "per_seed_data",
                                "sampled_step_details")}
    all_html = generate_html(top_all, bot_all, fixed_model="_all_", fixed_dataset="_agg_")
    all_path = os.path.join(out_dir, "all.html")
    with open(all_path, "w", encoding="utf-8") as f:
        f.write(all_html)
    size_kb = os.path.getsize(all_path) / 1024
    print(f"  [All] all.html ({size_kb:.0f} KB)")

    # Generate index page
    idx_html = _generate_index_html(models, datasets, top_data)
    idx_path = os.path.join(out_dir, "index.html")
    with open(idx_path, "w", encoding="utf-8") as f:
        f.write(idx_html)
    print(f"\n[OK] Generated {len(all_keys)} pages + all.html + index.html in {out_dir}/")


if __name__ == "__main__":
    main()
