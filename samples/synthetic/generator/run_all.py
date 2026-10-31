import json, sys, pandas as pd
from pathlib import Path
sys.path.insert(0, "samples_v2/generator"); import evaluator as ev
root = Path("samples_v2"); gt = pd.read_csv(root/"ground_truth.csv")
rows = []
for lab in ["spam_detection","customer_segments","news_topics"]:
    rub = json.loads((root/lab/"rubric.json").read_text())
    for p in sorted((root/lab).glob("*.ipynb")):
        r = ev.evaluate(p, rub); r["lab"]=lab; r["confidence"]=ev.confidence(r, rub)
        r["baseline_expl"]=ev.baseline_explanation(p, rub); r["improved_expl"]=round(100*r["explanation (/20)"]/20)
        rows.append(r)
df = pd.DataFrame(rows).merge(gt, left_on=["lab","student"], right_on=["lab","submission"])
df.to_csv("results_v2.csv", index=False)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",70)
print(df[["student","correctness (/35)","output (/15)","errors (/15)","documentation (/15)","explanation (/20)","total (/100)","grade","expected_band","baseline_expl","improved_expl","confidence"]].to_string())
print(df[["student","should_flag","correct output, wrong explanation"]].to_string())
print(df[["student","marks lost"]].to_string())
