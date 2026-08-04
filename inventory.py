from pathlib import Path
from collections import Counter

root = Path("data/external/sturm_fusion_24/extracted")

files = [p for p in root.rglob("*") if p.is_file()]

print("="*100)
print("TOTAL FILES")
print("="*100)
print(len(files))

print("\nTOP LEVEL")
print("="*100)
for p in sorted(root.iterdir()):
    print(p)

print("\nEXTENSIONS")
print("="*100)
for k, v in Counter(f.suffix.lower() for f in files).most_common():
    print(f"{k:12s} {v}")

print("\nFIRST 200 FILES")
print("="*100)
for f in files[:200]:
    print(f.relative_to(root))
