# Jupyter Notebook Lab Evaluator

Group 3, Statistical NLP: S016, S017, S020, S025, S026, S027

This tool grades student lab notebooks (`.ipynb`). It looks at both the code and the written explanation, and uses classic NLP methods (TF-IDF, Sentence-BERT similarity, rule-based algorithm detection) rather than an LLM. For each notebook it gives:

- marks for correctness (35), expected output (15), errors (15), documentation (15) and explanation (20)
- a total out of 100 and a grade (A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, F below that)
- one line saying where marks were lost
- a flag when the code works but the explanation describes a different algorithm (for example, the code stems but the text says it lemmatizes)
- a confidence score, so a teacher knows which marks to check

## Repository layout

```
lab_evaluator.ipynb     main notebook: run it top to bottom
evaluator.py            the same logic as an importable module
requirements.txt
docs/API.md             function reference and how to add a new lab
samples/
  real/                 student notebooks + rubric.json (text_classification, kmeans)
  synthetic/            18 generated notebooks across 3 labs, ground_truth.csv, generator
results/                results_real.csv, results_v2.csv (synthetic)
demo/                   Streamlit app that shows the pipeline step by step
report/                 project report (.docx), slides (.pptx), figures
```

## Quick start

```bash
git clone https://github.com/manasdabhade/Jupyter-Notebook-Lab-Evaluator.git
cd Jupyter-Notebook-Lab-Evaluator
pip install -r requirements.txt
jupyter notebook lab_evaluator.ipynb
```

Run all cells. By default it grades `samples/real/text_classification`. To grade a different lab, change `RUBRIC` and `NOTEBOOKS` in Section 9. In Google Colab, choose Runtime → Run all and upload the rubric and the notebooks when asked.

Output goes to `reports/`: `gradebook.csv`, `results.json` and `mark_distribution.png`.

## Demo app

```bash
cd demo
pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a notebook (or pick one of the bundled samples) and watch it move through the seven stages: input, parse, execute, detect, compare, score, report.

## Results

| Set | Notebooks | Flag precision | Flag recall |
|---|---|---|---|
| Real | 5 | 1.00 (2 of 2) | 1.00 |
| Synthetic | 18 | 0.50 | 1.00 |
| Combined | 23 | 0.63 | 1.00 |

The improved explanation score drops a wrong-algorithm explanation from 56 to 9, compared with the keyword-only baseline. Full details are in `report/`.

## Limitations

- When a student mentions an alternative they rejected ("we did not use Naive Bayes"), it is still counted as a claim.
- Plain-language explanations that avoid technical terms score low.
- Algorithms written from scratch are not recognised.
- Student code is executed, so run the tool in Colab or a container.

Treat the mark as a first pass. A teacher should review flagged notebooks.
