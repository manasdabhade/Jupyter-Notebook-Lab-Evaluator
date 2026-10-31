"""Generate the new synthetic sample set: 3 labs x 6 submissions, plus rubrics and ground truth."""
import json, csv, copy, subprocess, sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code
from nbclient import NotebookClient

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "samples_v2")

# ---------------------------------------------------------------- lab data
SPAM = [("win a free prize now", 1), ("claim your cash reward today", 1), ("free entry to win a holiday", 1),
        ("urgent you have won a gift card", 1), ("call now to claim free cash", 1), ("exclusive offer win money now", 1),
        ("congratulations you won a free phone", 1), ("cheap loans approved click now", 1), ("limited offer free prize inside", 1),
        ("reply win to get your cash prize", 1), ("you are selected for a free reward", 1), ("claim free bonus money today", 1),
        ("winner claim your prize now", 1), ("get free cash with this offer", 1), ("act now free gift waiting", 1),
        ("are we still meeting for lunch", 0), ("can you send me the notes", 0), ("see you at the library at five", 0),
        ("happy birthday hope you have a great day", 0), ("running late will call you soon", 0), ("did you finish the assignment", 0),
        ("lets watch a movie tonight", 0), ("mom asked you to call her", 0), ("the lecture moved to room four", 0),
        ("thanks for dinner yesterday", 0), ("can we meet tomorrow morning", 0), ("i left my charger at your place", 0),
        ("what time is the match", 0), ("please bring the lab report", 0), ("good luck in the exam", 0)]

NEWS = [("striker scores twice as united win the derby", "sport"), ("coach praises defence after cup victory", "sport"),
        ("tennis champion reaches the final in straight sets", "sport"), ("cricket team announces squad for test series", "sport"),
        ("goalkeeper saves penalty in late drama", "sport"), ("olympic sprinter breaks national record", "sport"),
        ("league title race goes to the final match", "sport"), ("captain injured ahead of world cup qualifier", "sport"),
        ("central bank raises interest rates again", "business"), ("shares fall as quarterly profits miss forecasts", "business"),
        ("startup raises funding to expand into europe", "business"), ("inflation slows for the third month", "business"),
        ("retailer reports record holiday sales", "business"), ("oil prices climb on supply concerns", "business"),
        ("bank announces merger with regional lender", "business"), ("stock market closes at record high", "business"),
        ("new smartphone launches with faster chip", "tech"), ("researchers unveil open source language model", "tech"),
        ("software update fixes security flaw in browser", "tech"), ("chipmaker reveals next generation processor", "tech"),
        ("app store policy changes anger developers", "tech"), ("cloud outage disrupts popular websites", "tech"),
        ("robotics company demos warehouse robot", "tech"), ("new laptop promises longer battery life", "tech")]

COMMON_CLASSIFY = '''def preprocess(text):
    """Lower-case a message and strip punctuation."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ")  # keep letters, digits, spaces
'''

# ---------------------------------------------------------------- lab definitions
LABS = {}

# Lab 1: SMS spam, bag of words + logistic regression
LABS["spam_detection"] = dict(
    prefix="SP",
    title="# Lab: SMS spam detection",
    data=f"data = {SPAM!r}  # (message, label) pairs; 1 = spam\ntexts = [t for t, _ in data]\nlabels = [y for _, y in data]\nprint(len(texts), 'messages,', sum(labels), 'spam')",
    steps=[
        ("## Step 1: Clean the text", COMMON_CLASSIFY + "clean = [preprocess(t) for t in texts]  # normalised messages\nprint(clean[:3])"),
        ("## Step 2: Bag-of-words features",
         "from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.model_selection import train_test_split\n"
         "vectorizer = CountVectorizer()  # raw word counts\nX = vectorizer.fit_transform(clean)\n"
         "X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=1, stratify=labels)  # hold out 30%\n"
         "print('vocabulary size:', len(vectorizer.vocabulary_))"),
        ("## Step 3: Train logistic regression",
         "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\n"
         "model = LogisticRegression(max_iter=1000)  # linear model with a sigmoid output\nmodel.fit(X_train, y_train)\n"
         "print('Accuracy:', round(accuracy_score(y_test, model.predict(X_test)), 2))"),
        ("## Step 4: Try a new message",
         "msg = 'win a free cash prize'  # unseen example\nprint('Prediction:', 'spam' if model.predict(vectorizer.transform([preprocess(msg)]))[0] else 'ham')"),
    ],
    good=("Each message is lower-cased and stripped of punctuation, then turned into a bag of words: CountVectorizer builds a "
          "vocabulary of every word in the training messages and represents each message as a vector of word counts, ignoring "
          "word order. We hold out 30% of the messages as a test set. A logistic regression classifier learns one weight per "
          "vocabulary word; the weighted sum is passed through the sigmoid function to give the probability that a message is "
          "spam, and messages above 0.5 are labelled spam. Words such as free, win and prize get large positive weights. "
          "Accuracy on the held-out test set measures how often the predicted label matches the true label, and the model "
          "correctly flags the new message about a free cash prize as spam."),
    wrong=("Each message is first lemmatized with the WordNet lemmatizer so that words like winning and won become win. We then "
           "compute TF-IDF weights, where each word is weighted by its term frequency times its inverse document frequency, so "
           "rare words count more than common ones. A multinomial Naive Bayes classifier then estimates the probability of each "
           "word given the class and applies Bayes rule, assuming the words are independent, to pick the most likely label. "
           "Accuracy on the held-out test set shows how often the prediction is correct."),
    negation=None,  # filled below
    paraphrase=("We tidy every message by making it lower case and removing punctuation. Then we count how many times each "
                "word appears, so every message becomes a list of numbers over all the words seen in training, and the "
                "order of words is thrown away. Part of the data is kept aside to check the model later. The model learns "
                "how strongly each word pushes a message towards being junk, adds these up and squashes the total into a "
                "number between zero and one. Messages full of words like free and prize end up near one. Finally we check "
                "what share of the kept-aside messages it gets right."),
    broken_typo=("clean", "cleaned"),
    rubric={
        "reference": "Messages are cleaned and converted to a bag of words with CountVectorizer, which builds a vocabulary "
                     "and represents each message as word counts. The data is split into training and test sets. A logistic "
                     "regression classifier learns a weight for each word and applies the sigmoid function to output the "
                     "probability of spam. Accuracy on the test set measures performance.",
        "concepts": {"text cleaning": ["lower", "punctuation", "clean"], "bag of words": ["bag of words", "word count", "countvectorizer"],
                     "vocabulary": ["vocabulary"], "train/test split": ["test set", "held-out", "hold out", "split"],
                     "logistic regression": ["logistic regression", "sigmoid"], "evaluation": ["accuracy"]},
        "expected": [{"name": "accuracy >= 0.75", "pattern": "accuracy\\D*([0-9.]+)", "min": 0.75},
                     {"name": "new message flagged as spam", "contains": "Prediction: spam"}],
        "tests": {"model is logistic regression": "assert type(model).__name__ == 'LogisticRegression'",
                  "features are bag of words": "assert type(vectorizer).__name__ == 'CountVectorizer'",
                  "detects obvious spam": "assert model.predict(vectorizer.transform(['claim your free prize now']))[0] == 1"},
    },
)
LABS["spam_detection"]["negation"] = LABS["spam_detection"]["good"] + (
    " We did not use Naive Bayes; we chose logistic regression instead because its weights are easy to read.")

# Lab 2: customer segmentation, hierarchical clustering
LABS["customer_segments"] = dict(
    prefix="CS",
    title="# Lab: Customer segmentation",
    data="from sklearn.datasets import make_blobs\nimport numpy as np\n"
         "X, _ = make_blobs(n_samples=150, centers=[[20, 30], [45, 70], [70, 25]], cluster_std=5, random_state=7)  # (age, spend score)\n"
         "print(X.shape)",
    steps=[
        ("## Step 1: Scale the features",
         "from sklearn.preprocessing import StandardScaler\nX_scaled = StandardScaler().fit_transform(X)  # mean 0, variance 1\n"
         "print(X_scaled.mean(axis=0).round(2))"),
        ("## Step 2: Agglomerative clustering",
         "from sklearn.cluster import AgglomerativeClustering\n"
         "model = AgglomerativeClustering(n_clusters=3, linkage='ward')  # merge clusters bottom-up\nlabels = model.fit_predict(X_scaled)\n"
         "print(len(set(labels)), 'clusters')"),
        ("## Step 3: Evaluate",
         "from sklearn.metrics import silhouette_score\nprint('Silhouette:', round(silhouette_score(X_scaled, labels), 2))  # -1 to 1"),
        ("## Step 4: Describe the segments",
         "def segment_means(X, labels):\n    \"\"\"Average age and spend score of each cluster.\"\"\"\n"
         "    return {int(k): X[labels == k].mean(axis=0).round(1).tolist() for k in sorted(set(labels))}\n"
         "print(segment_means(X, labels))"),
    ],
    good=("The two features, age and spending score, are standardised so both have mean 0 and variance 1 and neither dominates "
          "the distance. We then use agglomerative hierarchical clustering: every customer starts as its own cluster and, at "
          "each step, the two closest clusters are merged. Ward linkage chooses the merge that least increases the total "
          "within-cluster variance. Repeating this builds a tree of merges, a dendrogram, and cutting the tree at three "
          "clusters gives the segments. The silhouette score compares how close each point is to its own cluster versus the "
          "nearest other cluster; values near 1 mean well separated segments. The segment averages describe a young low "
          "spending group, a middle aged high spending group and an older low spending group."),
    wrong=("We scale the features and then run K-Means clustering. K-Means picks three random centroids, assigns each customer "
           "to the nearest centroid, and then moves each centroid to the mean of its assigned points. These two steps repeat "
           "until the centroids stop moving, which minimises the within-cluster sum of squares. The silhouette score shows the "
           "clusters are well separated and the segment averages describe three customer groups."),
    negation=None,
    paraphrase=("Age and spending are put on the same scale so one does not outweigh the other. Each customer begins in a group "
                "of one, and the two nearest groups are joined again and again, always choosing the join that keeps groups "
                "tight. Stopping when three groups remain gives the segments. A score between minus one and one tells us how "
                "much closer customers are to their own group than to the next group, and higher is better. The group averages "
                "describe who is in each segment."),
    broken_typo=("X_scaled", "x_scaled"),
    rubric={
        "reference": "The features are standardised. Agglomerative hierarchical clustering starts with every point as its own "
                     "cluster and repeatedly merges the closest clusters; Ward linkage merges the pair that least increases "
                     "within-cluster variance. The resulting dendrogram is cut at three clusters. The silhouette score "
                     "measures how well separated the clusters are.",
        "concepts": {"scaling": ["standardis", "standardiz", "scale"], "bottom-up merging": ["merge", "merged", "joined"],
                     "linkage": ["linkage", "ward"], "dendrogram": ["dendrogram", "tree"],
                     "number of clusters": ["three clusters", "3 clusters", "three groups"], "silhouette": ["silhouette"]},
        "expected": [{"name": "silhouette >= 0.5", "pattern": "silhouette\\D*([0-9.]+)", "min": 0.5},
                     {"name": "three clusters", "contains": "3 clusters"}],
        "tests": {"model is agglomerative": "assert type(model).__name__ == 'AgglomerativeClustering'",
                  "three clusters found": "assert len(set(labels)) == 3",
                  "features scaled": "assert abs(X_scaled.mean()) < 1e-6"},
    },
)
LABS["customer_segments"]["negation"] = LABS["customer_segments"]["good"] + (
    " Unlike K-Means, this method does not need random starting centroids, so the result is the same every run.")

# Lab 3: news topics, TF-IDF + linear SVM
LABS["news_topics"] = dict(
    prefix="NT",
    title="# Lab: News headline topic classification",
    data=f"data = {NEWS!r}  # (headline, topic)\nheadlines = [h for h, _ in data]\ntopics = [t for _, t in data]\nprint(len(headlines), 'headlines')",
    steps=[
        ("## Step 1: TF-IDF features",
         "from sklearn.feature_extraction.text import TfidfVectorizer\n"
         "vectorizer = TfidfVectorizer(stop_words='english')  # weight words by rarity\nX = vectorizer.fit_transform(headlines)\nprint(X.shape)"),
        ("## Step 2: Train a linear SVM",
         "from sklearn.svm import LinearSVC\nfrom sklearn.model_selection import cross_val_score\n"
         "model = LinearSVC()  # maximum-margin linear classifier\n"
         "scores = cross_val_score(model, X, topics, cv=4)  # 4-fold cross-validation\n"
         "print('Mean accuracy:', round(scores.mean(), 2))\nmodel.fit(X, topics)"),
        ("## Step 3: Classify new headlines",
         "new = ['striker scores in cup final', 'bank cuts rates', 'chip shortage hits phones']  # unseen\n"
         "for h, p in zip(new, model.predict(vectorizer.transform(new))):\n    print(h, '->', p)"),
    ],
    good=("Each headline is converted into TF-IDF features. Term frequency counts how often a word appears in a headline, and "
          "inverse document frequency down-weights words that appear in many headlines, so distinctive words such as striker "
          "or inflation get high weights; English stop words are removed. We then train a linear support vector machine, "
          "which finds the hyperplane that separates the topics with the largest margin, using a one-vs-rest scheme for the "
          "three topics. Four-fold cross-validation trains on three quarters of the data and tests on the rest, four times, "
          "and the mean accuracy estimates performance on unseen headlines. Finally the model labels three new headlines."),
    wrong=("Each headline is turned into a bag of words, where we simply count how many times each word occurs. We then train "
           "a Naive Bayes classifier, which uses Bayes rule and the probability of each word within each topic, assuming the "
           "words are independent given the topic. Cross-validation estimates the accuracy and the model labels three new "
           "headlines."),
    negation=None,
    paraphrase=("Every headline becomes a list of numbers, one per word, where a word scores higher when it is frequent in this "
                "headline but rare across all headlines, and very common filler words are dropped. The classifier then draws "
                "the straight boundary that keeps the topics as far apart as possible. We test it by rotating which quarter of "
                "the data is held back, four times, and averaging how often it is right. It then guesses the topic of three "
                "new headlines."),
    broken_typo=("X, topics", "X, topic"),
    rubric={
        "reference": "Headlines are represented with TF-IDF, which multiplies term frequency by inverse document frequency so "
                     "rare, distinctive words get higher weight; stop words are removed. A linear support vector machine "
                     "finds the maximum-margin hyperplane separating the topics, one-vs-rest. Cross-validation estimates "
                     "accuracy on unseen data.",
        "concepts": {"tf-idf": ["tf-idf", "inverse document frequency"], "stop words": ["stop word"],
                     "svm": ["support vector", "svm"], "margin": ["margin"], "cross-validation": ["cross-validation", "fold"],
                     "accuracy": ["accuracy"]},
        "expected": [{"name": "mean accuracy >= 0.6", "pattern": "mean accuracy\\D*([0-9.]+)", "min": 0.6},
                     {"name": "sport headline classified", "contains": "striker scores in cup final -> sport"}],
        "tests": {"features are TF-IDF": "assert type(vectorizer).__name__ == 'TfidfVectorizer'",
                  "model is a linear SVM": "assert type(model).__name__ == 'LinearSVC'",
                  "three topics learned": "assert len(model.classes_) == 3"},
    },
)
LABS["news_topics"]["negation"] = LABS["news_topics"]["good"] + (
    " We considered Naive Bayes, but compared to Naive Bayes the SVM handles correlated words better.")

# ---------------------------------------------------------------- variants
VARIANTS = [
    # (suffix, designed as, explanation key, should flag, expected band)
    ("01", "Strong submission: correct code, full explanation", "good", False, "A"),
    ("02", "Correct code; explanation describes a different algorithm", "wrong", True, "B-C"),
    ("03", "Typo causing runtime errors, a cell never run, one-line explanation", "short_broken", False, "F"),
    ("04", "Correct code; explanation is one sentence", "short", False, "B-C"),
    ("05", "Correct code and explanation, plus a sentence naming a rejected alternative", "negation", False, "A"),
    ("06", "Correct code; explanation paraphrased in plain words with few technical terms", "paraphrase", False, "A-B"),
]
SHORT = "The code works and gives the right answer."


def build(lab, v):
    suffix, _, key, _, _ = v
    cells = [md(lab["title"]), md("## Load the data"), code(lab["data"])]
    for i, (head, src) in enumerate(lab["steps"]):
        if key == "short_broken" and i == 1:
            a, b = lab["broken_typo"]
            src = src.replace(a, b, 1) if a in src else src
        cells += [md(head), code(src)]
    text = {"good": lab["good"], "wrong": lab["wrong"], "negation": lab["negation"],
            "paraphrase": lab["paraphrase"], "short": SHORT, "short_broken": "It classifies the data."}[key]
    cells += [md("## Explanation\n" + text)]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


rows = []
for name, lab in LABS.items():
    d = ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "rubric.json").write_text(json.dumps(lab["rubric"], indent=2))
    for v in VARIANTS:
        nb = build(lab, v)
        NotebookClient(nb, timeout=120, allow_errors=True, kernel_name="lab-evaluator").execute()
        if v[2] == "short_broken":       # leave the last code cell un-run, as a student would
            last = [c for c in nb.cells if c.cell_type == "code"][-1]
            last.execution_count, last.outputs = None, []
        sid = f"{lab['prefix']}-{v[0]}"
        nbformat.write(nb, d / f"{sid}.ipynb")
        rows.append({"lab": name, "submission": sid, "designed_as": v[1], "should_flag": v[3], "expected_band": v[4]})

with open(ROOT / "ground_truth.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} notebooks to {ROOT}")
