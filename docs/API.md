# Lab Evaluator API

The evaluator is the notebook `lab_evaluator.ipynb`. Run sections 1 to 8 (or Runtime → Run all) and these functions are available. The same logic is in `evaluator.py`.

## evaluate(path, rubric={}) → dict

Grades one student notebook.

* `path`: path to the student's `.ipynb`.
* `rubric`: the lab's rubric as a dict (see section 2 of the notebook). Every key is optional: `reference`, `concepts`, `expected`, `tests`.

Returns one gradebook row:

| Key | Meaning |
| --- | --- |
| `student` | file name without `.ipynb` |
| `correctness (/35)`, `output (/15)`, `errors (/15)`, `documentation (/15)`, `explanation (/20)` | marks per check; a check the rubric does not define is left out and the rest are scaled to 100 |
| `total (/100)`, `grade` | overall mark and grade (A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, else F) |
| `marks lost` | one short reason per check that lost marks, e.g. `explanation -16.0: describes a different algorithm than the code` |
| `correct output, wrong explanation` | e.g. `says lemmatization, code uses stemming`; empty if not flagged |

## to_json(row) → dict

Turns a row into the structured record written to `reports/results.json`:

```json
{
  "input_id": "S016",
  "result": {
    "marks": {"correctness": 35.0, "output": 15.0, "errors": 15.0, "documentation": 13.9, "explanation": 20.0},
    "total": 98.9,
    "grade": "A",
    "marks_lost": "documentation -1.1: too little written explanation",
    "wrong_explanation": "",
    "confidence": 0.9
  }
}
```

`confidence` (0 to 1) starts at 1 and drops by 0.1 when only TF-IDF similarity is available, 0.2 when the rubric has no tests (or `RUN_CODE` is off), 0.2 when the notebook is flagged, and 0.1 when the total is within 2 marks of a grade boundary. Records below about 0.7 should be checked by a teacher.

## Files written

* `reports/gradebook.csv`: one row per student (the table above).
* `reports/results.json`: list of `to_json` records.
* `reports/mark_distribution.png`: histogram of totals.

## Settings (section 1)

* `RUN_CODE`: re-run student notebooks and rubric tests (default `True`).
* `WEIGHTS`: marks per check, out of 100.

## Supporting a new lab

Write a `rubric.json` for it. If the lab uses an algorithm not in the `ALGORITHMS` table (section 5), add one line for it.
