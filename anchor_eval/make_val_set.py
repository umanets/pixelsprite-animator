"""Writes val_chars.json: held-out (split "val") characters of the 3D farm with their ORIGINAL v2 sprite tokens.

Read-only on pilot3d/. Run from the repo root with the repo venv:
    .venv\Scripts\python.exe animator_ui\anchor_eval\make_val_set.py [--per 8,4,4]
Picks, per source size (16 / 32) and kind (human, quadruped, bird), evenly spaced characters of the sorted val names
that are finished in a farm folder. The ground-truth renders are read by the selftest from the farm folders.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--per", default="8,4,4", help="characters per size for human,quadruped,bird")
a = ap.parse_args()
per = dict(zip(["human", "quadruped", "bird"], [int(x) for x in a.per.split(",")]))
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
o = np.load(os.path.join(root, "pilot3d", "train", "orig_tokens.npz"), allow_pickle=True)
names, split = o["names"].tolist(), o["split"].tolist()
tok = {}
for s in (16, 32):
    for n, t in zip(o[f"names{s}"].tolist(), o[f"t{s}"]):
        tok[(n, s)] = t
farms = [os.path.join(root, "pilot3d", d) for d in ("farm_out_trial", "farm_out_test", "farm_out_full")]
cands = {}
for n, sp in sorted(zip(names, split)):
    if sp != "val":
        continue
    d = next((os.path.join(f, n) for f in farms if os.path.isfile(os.path.join(f, n, "pix", "sprites.npz"))), None)
    if d is None:
        continue
    try:
        kind = json.load(open(os.path.join(d, "job.json")))["kind"]
        caption = json.load(open(os.path.join(d, "job.json"))).get("caption", "")
    except Exception:
        continue
    for s in (16, 32):
        if (n, s) in tok and kind in per:
            cands.setdefault((s, kind), []).append((n, caption))
out = []
for (s, kind), lst in sorted(cands.items()):
    k = per[kind]
    idx = np.linspace(0, len(lst) - 1, k).round().astype(int) if len(lst) > k else range(len(lst))
    for i in sorted(set(int(i) for i in idx)):
        n, cap = lst[i]
        out.append({"name": n, "kind": kind, "size": s, "caption": cap, "tokens": [int(x) for x in tok[(n, s)]]})
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "val_chars.json"), "w"))
print(len(out), "characters", {f"{s}/{k}": len([x for x in out if x['size'] == s and x['kind'] == k]) for (s, k) in cands})
