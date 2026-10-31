# Sample set v2 (final report experiment)

Synthetic test set for the Lab Evaluator: 3 labs x 6 submissions = 18 notebooks.
Submission IDs (SP-, CS-, NT-) are anonymous and unrelated to any student or team roll number.
Every notebook was executed before saving, so outputs are present exactly as a student would submit them.

| Folder | Lab | Method the code uses |
| --- | --- | --- |
| spam_detection/ | SMS spam detection (30 messages) | Bag of words (CountVectorizer) + logistic regression |
| customer_segments/ | Customer segmentation (150 synthetic points) | Standard scaling + agglomerative (hierarchical) clustering |
| news_topics/ | News headline topics (24 headlines, 3 topics) | TF-IDF + linear SVM |

Each folder contains rubric.json and six submissions built the same way:

| Suffix | Designed as | Should be flagged | Expected grade band |
| --- | --- | --- | --- |
| -01 | Strong submission: correct code, full explanation | No | A |
| -02 | Correct code; explanation describes a different algorithm | Yes | B-C |
| -03 | Typo causing runtime errors, last cell never run, one-line explanation | No | F |
| -04 | Correct code; one-sentence explanation | No | B-C |
| -05 | Correct code and explanation, plus one sentence naming a rejected alternative ("we did not use Naive Bayes") | No | A |
| -06 | Correct code; explanation paraphrased in plain words | No | A-B |

ground_truth.csv lists the design of every notebook. results_v2.csv is the evaluator's output on all 18.

To grade one lab in lab_evaluator.ipynb (Section 9):
RUBRIC = "samples_v2/spam_detection/rubric.json"; NOTEBOOKS = "samples_v2/spam_detection/*.ipynb"

generator/make_samples.py rebuilds the whole set; generator/run_all.py grades it.
