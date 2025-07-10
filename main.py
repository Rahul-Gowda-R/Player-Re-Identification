import os

print(">> RUNNING PIPELINE <<\n")

print("[1] Running detection...")
os.system("python detect.py")

print("[2] Extracting features...")
os.system("python extract_features.py")

if os.path.exists("data/tacticam_tracking.json"):
    print("[3] Matching players across views...")
    os.system("python match_players.py")
else:
    print("⚠️ tacticam data not found. skipping step 3")

print("\n[✅] all steps done.")

