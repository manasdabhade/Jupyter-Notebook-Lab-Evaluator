"""Lab Evaluator — interactive pipeline demo.  Run:  streamlit run app.py"""
import json, re, time, html, tempfile
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import nbformat
import evaluator as ev

st.set_page_config(page_title="Lab Evaluator · Live Demo", page_icon="◐", layout="wide")
HERE = Path(__file__).parent
CY, PU, OR, RED, GRN, GR = "#2B2622", "#B89C8D", "#C9A27E", "#A8473A", "#6F7F62", "#E9DED6"
INK, MUTED, BLUSH, SAND = "#2B2622", "#8C7F76", "#F3E6DF", "#D8C3B8"

# ------------------------------------------------------------------ styling + animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Jost:wght@300;400;500&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, button { font-family: 'Jost', 'Helvetica Neue', Arial, sans-serif; font-weight: 400; color:#2B2622; }
.block-container { padding-top: 2rem; max-width: 1180px; }
h1, h2, h3 { font-family: 'Cormorant Garamond', Georgia, serif !important; font-weight: 500 !important; letter-spacing: 0; color:#2B2622; }
h2 { font-size: 2.1rem !important; margin-top: 2.6rem !important; border-top: 1px solid #E9DED6; padding-top: 1.6rem !important; }
[data-testid="stSidebar"] { background:#FBF7F4; border-right: 1px solid #EFE6E0; }
[data-testid="stCaptionContainer"], .stCaption { color:#8C7F76 !important; }
.stButton button, .stDownloadButton button { border-radius: 0 !important; border: 1px solid #2B2622 !important; background: transparent !important; color:#2B2622 !important;
  letter-spacing: .14em; text-transform: uppercase; font-size: .75rem !important; padding: .7rem 1rem !important; }
.stButton button[kind="primary"] { background:#2B2622 !important; color:#F3E6DF !important; }
@keyframes fadeUp { from {opacity:0; transform: translateY(8px);} to {opacity:1; transform:none;} }
.fade { animation: fadeUp .6s ease both; }
.brand { font-family:'Jost',sans-serif; letter-spacing:.42em; font-weight:500; font-size:.95rem; color:#2B2622; }
.hero { background:#E8D6CC; background-image: linear-gradient(115deg, #C7AFA2 0 62%, #EBDCD3 62% 100%); padding: 4.2rem 3.4rem; margin: 1rem 0 1.6rem 0; }
.hero h1 { font-size: 3.6rem !important; color:#FFFFFF !important; margin:0 !important; line-height:1.05; }
.hero p { color:#FFFFFF; font-weight:300; font-size:1.05rem; max-width: 34rem; margin-top: .9rem; }
.stepper { display:flex; gap:1.6rem; margin:.2rem 0 1rem 0; border-bottom:1px solid #E9DED6; flex-wrap:wrap; }
.step { padding:.55rem 0; color:#B3A79F; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; border-bottom:1px solid transparent; margin-bottom:-1px; transition: all .4s; }
.step.done { color:#8C7F76; }
.step.active { color:#2B2622; border-bottom:1px solid #2B2622; }
.cellstrip { display:flex; gap:6px; flex-wrap:wrap; margin:.8rem 0 1.2rem 0; }
.cell { padding:.5rem .8rem; font-size:.72rem; letter-spacing:.06em; animation: fadeUp .5s ease both; border:1px solid #E9DED6; }
.md { background:#F3E6DF; } .code { background:#FFFFFF; } .unrun { background:#FFFFFF; border-style:dashed; color:#8C7F76; } .err { background:#FFFFFF; border-color:#A8473A; color:#A8473A; }
.test { display:flex; justify-content:space-between; padding:.6rem 0; border-bottom:1px solid #EFE6E0; animation: fadeUp .45s ease both; font-size:.92rem; }
.pass { color:#6F7F62; letter-spacing:.08em; font-size:.8rem; text-transform:uppercase; } .fail { color:#A8473A; letter-spacing:.08em; font-size:.8rem; text-transform:uppercase; }
.hl-code { border-bottom:1px solid #2B2622; }
.hl-bad { background:#F1D9D3; border-bottom:1px solid #A8473A; padding:0 2px; }
.hl-ok { background:#E7EADF; border-bottom:1px solid #6F7F62; padding:0 2px; }
.box { background:#FFFFFF; border:1px solid #EFE6E0; padding:1.2rem 1.4rem; line-height:1.8; font-weight:300; }
.label { font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; color:#8C7F76; margin-bottom:.4rem; }
.chip { display:inline-block; margin:.2rem .3rem .2rem 0; padding:.35rem .8rem; font-size:.78rem; border:1px solid #D8C3B8; animation: fadeUp .45s ease both; }
.chip.code { background:#FFFFFF; } .chip.text { background:#F3E6DF; } .chip.bad { background:#FFFFFF; border-color:#A8473A; color:#A8473A; }
.chip small { color:#8C7F76; }
.flag { border-left:2px solid #A8473A; background:#FBF1EE; padding:1.1rem 1.4rem; animation: fadeUp .6s ease both; }
.flag b { font-family:'Cormorant Garamond',Georgia,serif; font-size:1.5rem; font-weight:500; color:#A8473A; }
.okflag { border-left:2px solid #6F7F62; background:#F4F5EF; padding:1.1rem 1.4rem; animation: fadeUp .6s ease both; }
.grade { font-family:'Cormorant Garamond',Georgia,serif; font-size:5.4rem; text-align:center; line-height:1; animation: fadeUp .8s ease both; }
.big { font-family:'Cormorant Garamond',Georgia,serif; font-size:3.6rem; text-align:center; line-height:1.1; }
.sub { text-align:center; font-size:.7rem; letter-spacing:.18em; text-transform:uppercase; color:#8C7F76; }
.lost { padding:.55rem 0; border-bottom:1px solid #EFE6E0; animation: fadeUp .45s ease both; font-size:.92rem; }
.vs { display:grid; grid-template-columns: 1fr 40px 1fr; align-items:center; gap:.4rem; margin:.3rem 0; animation: fadeUp .45s ease both; }
.vs .arrow { text-align:center; color:#8C7F76; }
[data-testid="stMetricValue"] { font-family:'Cormorant Garamond',Georgia,serif; font-size:2.4rem !important; }
[data-testid="stMetricLabel"] p { font-size:.7rem !important; letter-spacing:.14em; text-transform:uppercase; color:#8C7F76 !important; }
</style>
""", unsafe_allow_html=True)

STAGES = ["01 Input", "02 Read", "03 Run", "04 Detect", "05 Compare", "06 Score", "07 Report"]


def stepper(active):
    cells = "".join(f'<div class="step {"done" if i < active else "active" if i == active else ""}">{s}</div>' for i, s in enumerate(STAGES))
    return f'<div class="stepper">{cells}</div>'


def pause(t=1.0):
    time.sleep(t * st.session_state.get("speed", 1.0))


def highlight(text, phrases, cls):
    out = html.escape(text)
    for p in sorted({p for ps in phrases.values() for p in ps}, key=len, reverse=True):
        out = re.sub(r"(?i)(\b" + re.escape(html.escape(p)) + r"\w*)", rf'<span class="{cls}">\1</span>', out)
    return out


def gauge(value, title, color, suffix="", maxv=100):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number={"suffix": suffix, "font": {"size": 40, "family": "Cormorant Garamond, Georgia, serif", "color": "#2B2622"}},
                                 title={"text": title.upper(), "font": {"size": 11, "color": "#8C7F76"}},
                                 gauge={"axis": {"range": [0, maxv]}, "bar": {"color": color, "thickness": .18}, "bgcolor": "#F3E6DF", "borderwidth": 0, "axis": {"range": [0, maxv], "tickcolor": "#D8C3B8", "tickfont": {"size": 10, "color": "#8C7F76"}}}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#2B2622", font_family="Jost, Arial")
    return fig


# ------------------------------------------------------------------ sidebar: input
with st.sidebar:
    st.markdown('<div class="brand">LAB&nbsp;EVALUATOR</div>', unsafe_allow_html=True)
    st.caption("Upload a lab notebook, or pick a sample, and watch every stage of the grading pipeline.")
    rubrics = {p.stem: json.loads(p.read_text()) for p in sorted((HERE / "rubrics").glob("*.json"))}
    SAMPLE_LAB = {"S016": "text_classification", "S017": "text_classification", "S020": "text_classification",
                  "S025": "kmeans", "S026": "kmeans", "SP": "spam_detection", "CS": "customer_segments", "NT": "news_topics"}
    mode = st.radio("Notebook source", ["Sample notebook", "Upload .ipynb"], horizontal=True)
    if mode == "Sample notebook":
        samples = sorted((HERE / "samples").glob("*.ipynb"))
        pick = st.selectbox("Sample", samples, format_func=lambda p: p.stem)
        default_lab = SAMPLE_LAB.get(pick.stem, SAMPLE_LAB.get(pick.stem[:2], "text_classification"))
        nb_path = pick
    else:
        up = st.file_uploader("Student notebook", type="ipynb")
        default_lab = "text_classification"; nb_path = None
        if up:
            tmp = Path(tempfile.mkdtemp()) / up.name; tmp.write_bytes(up.getvalue()); nb_path = tmp
    lab = st.selectbox("Lab rubric", list(rubrics), index=list(rubrics).index(default_lab))
    rub_up = st.file_uploader("…or upload your own rubric.json", type="json")
    rubric = json.loads(rub_up.getvalue()) if rub_up else rubrics[lab]
    st.session_state["speed"] = {"Slow (presenting)": 1.6, "Normal": 1.0, "Fast": 0.3}[st.select_slider("Animation speed", ["Slow (presenting)", "Normal", "Fast"], "Normal")]
    go_btn = st.button("Run the pipeline", type="primary", use_container_width=True, disabled=nb_path is None)
    with st.expander("Rubric being used"):
        st.json(rubric, expanded=False)

# ------------------------------------------------------------------ header
st.markdown('<div class="brand">LAB&nbsp;EVALUATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Code, and the<br>understanding behind it.</h1><p>A transparent grader for Jupyter lab notebooks. It runs the code, reads the explanation, and notices when the two describe different things.</p></div>', unsafe_allow_html=True)
bar = st.empty()
bar.markdown(stepper(0), unsafe_allow_html=True)

if not go_btn:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="box fade"><div class="label">Input</div>A student .ipynb + the lab rubric (model answer, concepts, expected outputs, tests).</div>', unsafe_allow_html=True)
    c2.markdown('<div class="box fade"><div class="label">Pipeline</div>Read → run safely with tests → detect algorithms → compare meaning → score five checks.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="box fade"><div class="label">Output</div>Marks, grade, one-line feedback, a wrong-explanation flag and a confidence score.</div>', unsafe_allow_html=True)
    st.caption("Choose a notebook in the sidebar and press Run. S017 and CS-02 show the wrong-explanation flag.")
    st.stop()

ev.ensure_kernel()
original = nbformat.read(nb_path, as_version=4)

# ------------------------------------------------------------------ ① INPUT
st.markdown("## Input")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f'<div class="box fade"><b>Notebook:</b> {nb_path.name}<br><b>Cells:</b> {len(original.cells)}<br><b>Lab rubric:</b> {lab}</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="box fade">' + "<br>".join([f"<b>Model answer:</b> {len(rubric.get('reference','').split())} words",
                f"<b>Concepts:</b> {', '.join(rubric.get('concepts', {}))}", f"<b>Expected outputs:</b> {len(rubric.get('expected', []))}",
                f"<b>Tests:</b> {len(rubric.get('tests', {}))}"]) + "</div>", unsafe_allow_html=True)
pause(1.2)

# ------------------------------------------------------------------ run the real pipeline (re-executes the notebook)
bar.markdown(stepper(1), unsafe_allow_html=True)
with st.status("Running the evaluator on this notebook (re-executing it in a fresh kernel)…", expanded=False) as status:
    t0 = time.time(); R = ev.evaluate_detailed(nb_path, rubric)
    status.update(label=f"Evaluation finished in {time.time()-t0:.1f}s — now walking through each stage", state="complete")
nb = R["parsed"]
show_text = re.sub(r"#+\s*", "", nb["text"]).strip()

# ------------------------------------------------------------------ ② READ
st.markdown("## Reading the notebook")
st.caption("An .ipynb is JSON: a list of cells. `read_notebook()` sorts every cell into explanation text, code, comments, outputs and errors.")
strip = st.empty(); cells_html = ""
for i, c in enumerate(original.cells):
    if c.cell_type == "markdown": cls, lab_ = "md", "Markdown"
    elif c.get("execution_count") is None: cls, lab_ = "unrun", "Code · not run"
    elif any(o.output_type == "error" for o in c.get("outputs", [])): cls, lab_ = "err", "Code · error"
    else: cls, lab_ = "code", "Code"
    cells_html += f'<div class="cell {cls}" style="animation-delay:{i*0.08}s">#{i+1} {lab_}</div>'
    strip.markdown(f'<div class="cellstrip">{cells_html}</div>', unsafe_allow_html=True); pause(0.18)
m = st.columns(5)
for col, (k, v) in zip(m, [("Explanation words", R["words"]), ("Code lines", sum(1 for l in nb["code"].splitlines() if l.strip())),
                            ("Comments", len(nb["comments"])), ("Errors", len(nb["errors"])), ("Unrun cells", nb["not_run"])]):
    col.metric(k, v); pause(0.15)
t1, t2, t3 = st.tabs(["Explanation", "Code", "Outputs & errors"])
t1.markdown(f'<div class="box">{html.escape(show_text) or "<i>(no explanation)</i>"}</div>', unsafe_allow_html=True)
t2.code(nb["code"].strip() or "# no code", language="python")
t3.code((nb["outputs"].strip() or "(no output)") + ("\n\n" + "\n".join(nb["errors"]) if nb["errors"] else ""))
pause(1.2)

# ------------------------------------------------------------------ ③ RUN
bar.markdown(stepper(2), unsafe_allow_html=True)
st.markdown("## Running it, with the rubric tests")
st.caption("`run_notebook()` appends each test as a try/assert cell, then executes the whole notebook in a fresh kernel in a separate process.")
c1, c2 = st.columns([1.1, 1])
with c1:
    st.code("\n\n".join(f"# test: {n}\ntry:\n    {t}\n    print('PASS', {n!r})\nexcept Exception:\n    print('FAIL', {n!r})" for n, t in R["tests"].items()) or "# rubric has no tests", language="python")
with c2:
    ph = st.empty(); rows = ""
    for n in R["tests"]:
        ok = n in R["passed"]
        rows += f'<div class="test"><span>{html.escape(n)}</span><span class="{"pass" if ok else "fail"}">{"Pass" if ok else "Fail"}</span></div>'
        ph.markdown(rows, unsafe_allow_html=True); pause(0.6)
    n_t = len(R["tests"]) or 1
    st.progress(len(R["passed"]) / n_t, text=f"{len(R['passed'])} of {len(R['tests'])} tests passed")
pause(1.0)

# ------------------------------------------------------------------ ④ DETECT
bar.markdown(stepper(3), unsafe_allow_html=True)
st.markdown("## What the code does, and what the text claims")
st.caption("A table of 15 algorithms in 5 families. Same family = alternatives (stem *or* lemmatize; K-Means *or* hierarchical). Text claiming A while code uses B → contradiction.")
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="label">Found in the code</div>', unsafe_allow_html=True)
    st.markdown("".join(f'<span class="chip code" style="animation-delay:{i*.15}s">{a} <small>({ev.ALGORITHMS[a][0]})</small></span>' for i, a in enumerate(sorted(R["code_algos"]))) or "<i>none</i>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="label">Claimed in the explanation</div>', unsafe_allow_html=True)
    bad = {t for t, _ in R["wrong"]}
    st.markdown("".join(f'<span class="chip {"bad" if a in bad else "text"}" style="animation-delay:{i*.15}s">{a} <small>({ev.ALGORITHMS[a][0]})</small></span>' for i, a in enumerate(sorted(R["text_algos"]))) or "<i>none</i>", unsafe_allow_html=True)
pause(0.8)
good_ph = {a: p for a, p in R["phrases"].items() if a in R["code_algos"]}
bad_ph = {a: p for a, p in R["phrases"].items() if a in bad}
hl = highlight(show_text, bad_ph, "hl-bad")
hl = re.sub("x^", "", hl)
for p in sorted({p for ps in good_ph.values() for p in ps}, key=len, reverse=True):
    hl = re.sub(r"(?i)(?<![\w>])(" + re.escape(html.escape(p)) + r"\w*)", r'<span class="hl-ok">\1</span>', hl)
st.markdown("<div class='label'>The explanation, annotated · <span class='hl-ok'>agrees with the code</span> · <span class='hl-bad'>contradicts it</span></div>", unsafe_allow_html=True)
st.markdown(f'<div class="box fade">{hl or "<i>(no explanation)</i>"}</div>', unsafe_allow_html=True)
pause(0.6)
if R["wrong"]:
    st.markdown('<div class="label">Same-family mismatches</div>', unsafe_allow_html=True)
    for t, c in R["wrong"]:
        st.markdown(f'<div class="vs"><div class="chip bad">text says: {t}</div><div class="arrow">≠</div><div class="chip code">code uses: {c}</div></div>', unsafe_allow_html=True); pause(0.5)
else:
    st.markdown('<div class="okflag">No contradiction. Every algorithm named in the text is consistent with the code.</div>', unsafe_allow_html=True)
pause(1.0)

# ------------------------------------------------------------------ ⑤ COMPARE
bar.markdown(stepper(4), unsafe_allow_html=True)
st.markdown("## Comparing with the model answer")
st.caption(f"Concept coverage (rubric keywords) + semantic similarity ({R['backend']}). Explanation score = 0.6 × coverage + 0.4 × similarity.")
c1, c2, c3 = st.columns([1.1, 1, 1])
with c1:
    st.markdown('<div class="label">Concept checklist</div>', unsafe_allow_html=True)
    box = st.empty(); acc = ""
    for c, hits in R["concept_hits"].items():
        acc += f'<div class="test"><span>{html.escape(c)}</span><span class="{"pass" if hits else "fail"}">{", ".join(hits[:2]) if hits else "Missing"}</span></div>'
        box.markdown(acc, unsafe_allow_html=True); pause(0.35)
with c2:
    st.plotly_chart(gauge(R["coverage"] * 100, "Concept coverage", INK, "%"), use_container_width=True)
with c3:
    st.plotly_chart(gauge(R["similarity"] * 100, f"Similarity · {R['backend']}", PU, "%"), use_container_width=True)
if R["tfidf_terms"]:
    terms = R["tfidf_terms"][::-1]
    fig = go.Figure([go.Bar(y=[t for t, _, _ in terms], x=[a for _, a, _ in terms], orientation="h", name="Student", marker_color=INK),
                     go.Bar(y=[t for t, _, _ in terms], x=[b for _, _, b in terms], orientation="h", name="Model answer", marker_color=SAND)])
    fig.update_layout(barmode="group", height=300, title=dict(text="SHARED TF-IDF TERMS", font=dict(size=11, color="#8C7F76")), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#2B2622", font_family="Jost, Arial", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
expl_raw = 0.6 * R["coverage"] + 0.4 * R["similarity"]
steps = [f"0.6 × {R['coverage']:.2f} + 0.4 × {R['similarity']:.2f} = **{expl_raw:.2f}**"]
if R["words"] < 30: steps.append("Under 30 words → × 0.5")
elif R["wrong"]: steps.append("Describes a different algorithm → × 0.2")
steps.append(f"Explanation score = **{R['results']['explanation'][0]:.2f}**   (keyword-only baseline would give {R['baseline']/100:.2f})")
for s_ in steps: st.markdown("— " + s_); pause(0.5)
pause(0.8)

# ------------------------------------------------------------------ ⑥ SCORE
bar.markdown(stepper(5), unsafe_allow_html=True)
st.markdown("## Five checks")
st.caption("Correctness 35 · Output 15 · Errors 15 · Documentation 15 · Explanation 20 — each check gives a score from 0 to 1 and a reason.")
names = list(R["components"]); chart = st.empty()
for k in range(1, 11):
    frac = k / 10
    fig = go.Figure()
    fig.add_bar(y=names, x=[R["components"][n]["out_of"] for n in names], orientation="h", marker_color="#F3E6DF", name="Available", hoverinfo="skip")
    fig.add_bar(y=names, x=[R["components"][n]["marks"] * frac for n in names], orientation="h",
                marker_color=[INK if R["components"][n]["score"] >= .9 else PU if R["components"][n]["score"] >= .5 else RED for n in names],
                text=[f'{R["components"][n]["marks"]:.1f} / {R["components"][n]["out_of"]:.0f}' if k == 10 else "" for n in names], textposition="outside", name="Earned")
    fig.update_layout(barmode="overlay", height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#2B2622", font_family="Jost, Arial", xaxis=dict(range=[0, 40]), margin=dict(l=10, r=40, t=10, b=10))
    chart.plotly_chart(fig, use_container_width=True, key=f"bars{k}"); time.sleep(0.07 * st.session_state["speed"])
c1, c2 = st.columns([1, 1.2])
with c1:
    rad = go.Figure(go.Scatterpolar(r=[R["components"][n]["score"] * 100 for n in names] + [R["components"][names[0]]["score"] * 100],
                                    theta=names + [names[0]], fill="toself", line_color=INK, fillcolor="rgba(184,156,141,.35)"))
    rad.update_layout(height=300, polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#E9DED6", tickfont=dict(size=9, color="#8C7F76")), angularaxis=dict(gridcolor="#E9DED6"), bgcolor="#FFFFFF"), paper_bgcolor="rgba(0,0,0,0)", font_color="#2B2622", font_family="Jost, Arial",
                      margin=dict(l=40, r=40, t=20, b=20), showlegend=False)
    st.plotly_chart(rad, use_container_width=True)
with c2:
    st.markdown('<div class="label">Where marks were lost</div>', unsafe_allow_html=True)
    if R["lost"]:
        for l in R["lost"]:
            st.markdown(f'<div class="lost">{html.escape(l)}</div>', unsafe_allow_html=True); pause(0.35)
    else:
        st.markdown('<div class="okflag">No marks lost.</div>', unsafe_allow_html=True)
pause(1.0)

# ------------------------------------------------------------------ ⑦ REPORT
bar.markdown(stepper(6), unsafe_allow_html=True)
st.markdown("## The result")
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    tot = st.empty()
    for v in [R["total"] * i / 20 for i in range(1, 21)]:
        tot.markdown(f'<div class="big">{v:.1f}<span style="font-size:1.4rem;color:#8C7F76"> / 100</span></div>', unsafe_allow_html=True); time.sleep(0.04 * st.session_state["speed"])
    st.markdown("<div class='sub'>Total mark</div>", unsafe_allow_html=True)
with c2:
    gcol = {"A": INK, "B": INK, "C": INK, "D": RED, "F": RED}[R["grade"]]
    st.markdown(f'<div class="grade" style="color:{gcol}">{R["grade"]}</div><div class="sub">Grade</div>', unsafe_allow_html=True)
with c3:
    st.plotly_chart(gauge(R["confidence"], "Confidence", INK if R["confidence"] > .7 else PU, "", 1), use_container_width=True)
pause(0.6)
if R["flag"]:
    st.markdown(f'<div class="flag"><b>Correct output, wrong explanation</b><br><span>{html.escape(R["flag"])}</span></div>', unsafe_allow_html=True)
elif R["wrong"] and not R["code_correct"]:
    st.markdown('<div class="okflag">No flag: the explanation mismatches the code, but the code itself is not correct — so the problem is the code.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="okflag">No wrong-explanation flag.</div>', unsafe_allow_html=True)
if R["confidence"] <= 0.7:
    st.caption("Confidence is 0.7 or lower, so this record would be sent to a teacher for review.")
record = {"input_id": nb_path.stem, "result": {"marks": {k: v["marks"] for k, v in R["components"].items()}, "total": R["total"], "grade": R["grade"],
          "marks_lost": "; ".join(R["lost"]) or "none", "wrong_explanation": R["flag"], "confidence": R["confidence"]}}
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown('<div class="label">results.json record</div>', unsafe_allow_html=True)
    st.json(record)
with c2:
    st.markdown('<div class="label">Download</div>', unsafe_allow_html=True)
    st.download_button("Download results.json", json.dumps(record, indent=2), f"{nb_path.stem}_results.json", "application/json", use_container_width=True)
    csv = "student,total,grade,marks_lost,flag,confidence\n" + f'{nb_path.stem},{R["total"]},{R["grade"]},"{record["result"]["marks_lost"]}","{R["flag"]}",{R["confidence"]}\n'
    st.download_button("Download gradebook row", csv, f"{nb_path.stem}_gradebook.csv", "text/csv", use_container_width=True)
bar.markdown(stepper(7), unsafe_allow_html=True)
