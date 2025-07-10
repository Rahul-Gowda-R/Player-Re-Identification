import json
import math
from collections import defaultdict

with open("data/broadcast_tracking.json", "r") as bf:
    broad_data = json.load(bf)

with open("data/tacticam_tracking.json", "r") as tf:
    tcam_data = json.load(tf)


DIST_THRESHOLD = 50  

scores = defaultdict(lambda: defaultdict(int))
counts = defaultdict(lambda: defaultdict(int))

for frame in broad_data:
    if frame not in tcam_data: continue

    b_players = broad_data[frame]
    t_players = tcam_data[frame]

    for b in b_players:
        b_id = b['id']
        bcx, bcy = b['centroid']

        min_dist = 1e6
        match_id = None

        for t in t_players:
            tid = t['id']
            tcx, tcy = t['centroid']

            d = ((bcx - tcx) ** 2 + (bcy - tcy) ** 2) ** 0.5
            if d < DIST_THRESHOLD and d < min_dist:
                min_dist = d
                match_id = tid

        if match_id is not None:
            scores[b_id][match_id] += min_dist
            counts[b_id][match_id] += 1


final_matches = {}

for b_id in scores:
    best_id = None
    lowest_avg = float('inf')

    for t_id in scores[b_id]:
        total = scores[b_id][t_id]
        c = counts[b_id][t_id]
        avg = total / c if c > 0 else float('inf')

        if avg < lowest_avg:
            lowest_avg = avg
            best_id = t_id

    if best_id is not None:
        final_matches[str(b_id)] = int(best_id)

import os
os.makedirs("outputs", exist_ok=True)

with open("outputs/matched_ids.json", "w") as out:
    json.dump(final_matches, out, indent=4)

print(f"[done] matched IDs → outputs/matched_ids.json")
