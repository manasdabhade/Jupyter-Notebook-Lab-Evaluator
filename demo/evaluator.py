"""Code cells of lab_evaluator.ipynb (Sections 1-8, 10, 12), unchanged in logic, as an importable module."""
import sys, subprocess, json, re, glob, tempfile
from pathlib import Path
import numpy as np, pandas as pd, nbformat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RUN_CODE = True
WEIGHTS = {"correctness": 35, "output": 15, "errors": 15, "documentation": 15, "explanation": 20}


def read_notebook(nb):
    if not isinstance(nb, nbformat.NotebookNode):
        nb = nbformat.read(nb, as_version=4)
    code, comments, text, outputs, errors, not_run = [], [], [], [], [], 0
    for cell in nb.cells:
        if cell.metadata.get("test"):
            continue
        if cell.cell_type == "markdown":
            text.append(re.sub(r"^# .*$", "", cell.source, flags=re.M))
        elif cell.cell_type == "code":
            for line in cell.source.splitlines():
                code.append(line.split("#")[0])
                if "#" in line:
                    comments.append(line.split("#", 1)[1].strip())
            if cell.get("execution_count") is None:
                not_run += 1
            for o in cell.get("outputs", []):
                if o.output_type == "error":
                    errors.append(f"{o.ename}: {o.evalue}")
                else:
                    outputs.append(o.get("text") or o.get("data", {}).get("text/plain", ""))
    return {"code": "\n".join(code), "comments": comments, "text": " ".join(text),
            "outputs": "\n".join(outputs), "errors": errors, "not_run": not_run,
            "code_cells": sum(c.cell_type == "code" for c in nb.cells),
            "functions": re.findall(r"def (\w+)\(.*\):\s*\n\s*(\"\"\"|''')?", "\n".join(c.source for c in nb.cells if c.cell_type == "code"))}


RUNNER = """
import sys, nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
NotebookClient(nb, timeout=60, allow_errors=True, kernel_name="lab-evaluator",
               resources={"metadata": {"path": sys.argv[3]}}).execute()
nbformat.write(nb, sys.argv[2])
"""


def run_notebook(path, tests):
    nb = nbformat.read(path, as_version=4)
    for name, test in tests.items():
        body = "\n".join("    " + l for l in test.splitlines())
        cell = nbformat.v4.new_code_cell(f"try:\n{body}\n    print('PASS', {name!r})\nexcept Exception:\n    print('FAIL', {name!r})")
        cell.metadata["test"] = True
        nb.cells.append(cell)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = f"{tmp}/in.ipynb", f"{tmp}/out.ipynb"
        nbformat.write(nb, src)
        subprocess.run([sys.executable, "-c", RUNNER, src, dst, str(Path(path).parent.resolve())], capture_output=True, timeout=600)
        nb = nbformat.read(dst, as_version=4)
    test_output = "".join(o.get("text", "") for c in nb.cells if c.metadata.get("test") for o in c.outputs)
    passed = [name for name in tests if f"PASS {name}" in test_output]
    return nb, passed


ALGORITHMS = {
    "stemming": ("normalisation", ["stemmer"], ["stem", "porter"]),
    "lemmatization": ("normalisation", ["lemmatiz"], ["lemmatiz", "lemma"]),
    "TF-IDF": ("features", ["tfidf"], ["tf-idf", "tfidf", "inverse document frequency"]),
    "bag of words": ("features", ["countvectorizer"], ["bag of words", "bag-of-words", "raw count"]),
    "word embeddings": ("features", ["word2vec", "sentencetransformer"], ["embedding", "word2vec"]),
    "Naive Bayes": ("classifier", ["naive_bayes", "nb("], ["naive bayes"]),
    "logistic regression": ("classifier", ["logisticregression"], ["logistic regression", "sigmoid"]),
    "SVM": ("classifier", ["svc("], ["support vector", "svm"]),
    "k-nearest neighbours": ("classifier", ["kneighbors"], ["nearest neighbo", "knn"]),
    "decision tree": ("classifier", ["decisiontree"], ["decision tree"]),
    "random forest": ("classifier", ["randomforest"], ["random forest"]),
    "K-Means": ("clustering", ["kmeans"], ["k-means", "kmeans"]),
    "hierarchical clustering": ("clustering", ["agglomerative", "linkage("], ["hierarchical", "agglomerative", "dendrogram"]),
    "DBSCAN": ("clustering", ["dbscan"], ["dbscan"]),
    "PCA": ("reduction", ["pca("], ["principal component", "pca"]),
}


def algorithms_in_code(code):
    code = code.lower()
    return {a for a, (_, words, _) in ALGORITHMS.items() if any(w in code for w in words)}


def algorithms_in_text(text):
    text = text.lower()
    return {a for a, (_, _, words) in ALGORITHMS.items() if any(re.search(r"\b" + re.escape(w), text) for w in words)}


def wrong_algorithms(code_algos, text_algos):
    return [(t, c) for t in text_algos - code_algos for c in code_algos if ALGORITHMS[t][0] == ALGORITHMS[c][0]]


KERNEL = "lab-evaluator"
try:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    sbert = None


def similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    if sbert:
        x, y = sbert.encode([a, b], normalize_embeddings=True)
        return float(np.clip((x @ y - 0.2) / 0.5, 0, 1))
    m = TfidfVectorizer(stop_words="english").fit_transform([a, b])
    return float(min(1, cosine_similarity(m[0], m[1])[0, 0] / 0.5))


def check_correctness(nb, passed, tests):
    if tests:
        failed = len(tests) - len(passed)
        return len(passed) / len(tests), f"{failed} of {len(tests)} tests failed" if failed else ""
    return max(0, 1 - len(nb["errors"]) / max(1, nb["code_cells"])), "some cells failed" if nb["errors"] else ""


def check_output(nb, expected):
    if not expected:
        return None, ""
    missing = 0
    for e in expected:
        if "contains" in e:
            ok = e["contains"].lower() in nb["outputs"].lower()
        else:
            numbers = [float(x) for x in re.findall(e["pattern"], nb["outputs"], re.I)]
            ok = any(e.get("min", -1e9) <= x <= e.get("max", 1e9) for x in numbers)
        missing += not ok
    return 1 - missing / len(expected), f"{missing} expected result(s) missing" if missing else ""


def check_errors(nb):
    score = max(0, 1 - 0.3 * len(nb["errors"]) - 0.1 * nb["not_run"])
    problems = ([f"{len(nb['errors'])} runtime error(s)"] if nb["errors"] else []) + (["cells not run"] if nb["not_run"] else [])
    return score, ", ".join(problems)


def check_documentation(nb):
    words = len(nb["text"].split())
    code_lines = max(1, sum(1 for l in nb["code"].splitlines() if l.strip()))
    with_doc = sum(1 for _, doc in nb["functions"] if doc)
    checks = {"too little written explanation": min(1, words / 150),
              "too few comments": min(1, len(nb["comments"]) / code_lines / 0.15),
              "functions lack docstrings": with_doc / len(nb["functions"]) if nb["functions"] else 1}
    return np.mean(list(checks.values())), ", ".join(k for k, v in checks.items() if v < 0.8)


def check_explanation(nb, rubric, wrong):
    text = nb["text"].lower()
    concepts = rubric.get("concepts", {})
    covered = [c for c, words in concepts.items() if any(w in text for w in words)]
    coverage = len(covered) / len(concepts) if concepts else 1
    score = 0.6 * coverage + 0.4 * similarity(nb["text"], rubric.get("reference", nb["code"]))
    if len(text.split()) < 30:
        return score * 0.5, "explanation too short"
    if wrong:
        return score * 0.2, "describes a different algorithm than the code"
    return score, "key concepts missing" if coverage < 1 else ("only partly matches the model answer" if score < 0.9 else "")


def grade(mark):
    return "A" if mark >= 90 else "B" if mark >= 80 else "C" if mark >= 70 else "D" if mark >= 60 else "F"


def evaluate(path, rubric={}):
    tests = rubric.get("tests", {})
    raw, passed = run_notebook(path, tests) if RUN_CODE else (nbformat.read(path, as_version=4), [])
    nb = read_notebook(raw)
    nb["not_run"] = read_notebook(path)["not_run"]
    code_algos, text_algos = algorithms_in_code(nb["code"]), algorithms_in_text(nb["text"])
    wrong = wrong_algorithms(code_algos, text_algos)
    results = {"correctness": check_correctness(nb, passed, tests if RUN_CODE else {}),
               "output": check_output(nb, rubric.get("expected")),
               "errors": check_errors(nb),
               "documentation": check_documentation(nb),
               "explanation": check_explanation(nb, rubric, wrong)}
    used = {k: w for k, w in WEIGHTS.items() if results[k][0] is not None}
    scale = 100 / sum(used.values())
    row = {"student": Path(path).stem}
    lost = []
    for k, w in used.items():
        score, reason = results[k]
        marks, out_of = score * w * scale, w * scale
        row[f"{k} (/{out_of:.0f})"] = round(marks, 1)
        if out_of - marks >= 0.5 and reason:
            lost.append(f"{k} -{out_of - marks:.1f}: {reason}")
    row["total (/100)"] = round(sum(v for k, v in row.items() if "(/" in k), 1)
    row["grade"] = grade(row["total (/100)"])
    row["marks lost"] = "; ".join(lost) or "none"
    output_ok = results["output"][0] is None or results["output"][0] >= 0.75
    code_correct = results["correctness"][0] >= 0.75 and output_ok
    row["correct output, wrong explanation"] = ", ".join(f"says {t}, code uses {c}" for t, c in wrong) if code_correct and wrong else ""
    return row


def confidence(row, rubric):
    c = 1.0
    if sbert is None:
        c -= 0.1
    if not (RUN_CODE and rubric.get("tests")):
        c -= 0.2
    if row["correct output, wrong explanation"]:
        c -= 0.2
    if min(abs(row["total (/100)"] - b) for b in (60, 70, 80, 90)) < 2:
        c -= 0.1
    return round(c, 2)


def baseline_explanation(path, rubric):
    text = read_notebook(path)["text"].lower()
    concepts = rubric.get("concepts", {})
    return round(100 * sum(any(w in text for w in ws) for ws in concepts.values()) / max(1, len(concepts)))


# ---------- helpers that expose intermediate results for the demo ----------
def ensure_kernel():
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "lab-evaluator"], capture_output=True)


def matched_phrases(text):
    """For highlighting: {algorithm: [phrases found in the text]}."""
    t = text.lower(); out = {}
    for a, (_, _, words) in ALGORITHMS.items():
        hits = [w for w in words if re.search(r"\b" + re.escape(w), t)]
        if hits: out[a] = hits
    return out


def matched_tokens(code):
    c = code.lower()
    return {a: [w for w in words if w in c] for a, (_, words, _) in ALGORITHMS.items() if any(w in c for w in words)}


def concept_hits(text, rubric):
    t = text.lower()
    return {c: [w for w in ws if w in t] for c, ws in rubric.get("concepts", {}).items()}


def top_tfidf_terms(a, b, k=8):
    v = TfidfVectorizer(stop_words="english"); m = v.fit_transform([a, b]).toarray()
    terms = v.get_feature_names_out(); shared = m[0] * m[1]
    idx = shared.argsort()[::-1][:k]
    return [(terms[i], float(m[0][i]), float(m[1][i])) for i in idx if shared[i] > 0]


def evaluate_detailed(path, rubric):
    """Same as evaluate(), but also returns every intermediate value the demo visualises."""
    tests = rubric.get("tests", {})
    original = nbformat.read(path, as_version=4)
    parsed_original = read_notebook(original)
    raw, passed = run_notebook(path, tests)
    nb = read_notebook(raw)
    nb["not_run"] = parsed_original["not_run"]
    code_algos, text_algos = algorithms_in_code(nb["code"]), algorithms_in_text(nb["text"])
    wrong = wrong_algorithms(code_algos, text_algos)
    results = {"correctness": check_correctness(nb, passed, tests),
               "output": check_output(nb, rubric.get("expected")),
               "errors": check_errors(nb),
               "documentation": check_documentation(nb),
               "explanation": check_explanation(nb, rubric, wrong)}
    used = {k: w for k, w in WEIGHTS.items() if results[k][0] is not None}
    scale = 100 / sum(used.values())
    comps, lost = {}, []
    for k, w in used.items():
        sc, reason = results[k]; out_of = w * scale; marks = sc * out_of
        comps[k] = {"score": float(sc), "marks": round(marks, 1), "out_of": round(out_of, 1), "reason": reason}
        if out_of - marks >= 0.5 and reason: lost.append(f"{k} -{out_of - marks:.1f}: {reason}")
    total = round(sum(c["marks"] for c in comps.values()), 1)
    output_ok = results["output"][0] is None or results["output"][0] >= 0.75
    code_correct = results["correctness"][0] >= 0.75 and output_ok
    flag = ", ".join(f"says {t}, code uses {c}" for t, c in wrong) if code_correct and wrong else ""
    row = {"student": Path(path).stem, "total (/100)": total, "grade": grade(total), "marks lost": "; ".join(lost) or "none",
           "correct output, wrong explanation": flag}
    conf = confidence(row, rubric)
    concepts = rubric.get("concepts", {}); hits = concept_hits(nb["text"], rubric)
    coverage = sum(1 for v in hits.values() if v) / len(concepts) if concepts else 1
    sim = similarity(nb["text"], rubric.get("reference", nb["code"]))
    return {"original": original, "parsed": nb, "tests": tests, "passed": passed, "code_algos": code_algos, "text_algos": text_algos,
            "wrong": wrong, "code_correct": code_correct, "results": results, "components": comps, "total": total, "grade": grade(total),
            "lost": lost, "flag": flag, "confidence": conf, "concept_hits": hits, "coverage": coverage, "similarity": sim,
            "baseline": baseline_explanation(path, rubric), "words": len(nb["text"].split()),
            "phrases": matched_phrases(nb["text"]), "tokens": matched_tokens(nb["code"]),
            "tfidf_terms": top_tfidf_terms(nb["text"], rubric.get("reference", "")) if rubric.get("reference") and nb["text"].strip() else [],
            "backend": "Sentence-BERT" if sbert else "TF-IDF"}
