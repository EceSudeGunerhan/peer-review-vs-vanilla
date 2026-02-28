# Run: python scripts/00_inspect_parsed_pdf.py 762
import json
import sys
from pathlib import Path

PARSED_DIR = Path("data/raw/iclr_2017/parsed_pdfs")

def short(v, n=200):
    s = str(v)
    s = s.replace("\n", " ")
    return s[:n] + ("..." if len(s) > n else "")

def main(paper_id: str):
    path = PARSED_DIR / f"{paper_id}.pdf.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    print("== Top-level keys ==")
    if isinstance(obj, dict):
        print(sorted(list(obj.keys()))[:60])
    else:
        print(type(obj))

    # metadata
    md = obj.get("metadata") if isinstance(obj, dict) else None
    print("\n== metadata keys ==")
    if isinstance(md, dict):
        print(sorted(list(md.keys())))

        for k in ["title", "abstract", "abstractText", "paper_text"]:
            if k in md:
                print(f"\nmetadata.{k} -> {short(md.get(k))}")
    else:
        print("metadata is not a dict / not present")

    # common section candidates
    candidates = [
        ("sections", obj.get("sections") if isinstance(obj, dict) else None),
        ("metadata.sections", md.get("sections") if isinstance(md, dict) else None),
        ("pdf_parse.sections", (obj.get("pdf_parse", {}) or {}).get("sections") if isinstance(obj, dict) else None),
        ("content.sections", (obj.get("content", {}) or {}).get("sections") if isinstance(obj, dict) else None),
        ("body_text", obj.get("body_text") if isinstance(obj, dict) else None),
    ]

    print("\n== section-like candidates ==")
    for name, val in candidates:
        if isinstance(val, list) and val:
            first = val[0]
            print(f"{name}: list(len={len(val)}) first_keys={list(first.keys()) if isinstance(first, dict) else type(first)}")
        else:
            t = type(val).__name__
            print(f"{name}: {t} (empty or missing)")

    # fallback: print any top-level keys that look like they might hold text
    print("\n== possible text fields (top-level) ==")
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            if any(x in k.lower() for x in ["text", "section", "content", "body", "paper", "abstract"]):
                v = obj.get(k)
                print(f"{k}: {type(v).__name__}")

if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "762"
    main(pid)