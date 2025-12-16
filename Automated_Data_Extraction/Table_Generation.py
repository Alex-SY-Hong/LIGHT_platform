import os
import re
import json
import csv
from collections import Counter

# ================= Configuration Area =================
FIELDS = [
    "Material Name",
    # Must add Polymer A/B/C/D/E fields
    "Polymer A Name", "Polymer A Type", "Polymer A Role",
    "Polymer B Name", "Polymer B Type", "Polymer B Role",
    "Polymer C Name", "Polymer C Type", "Polymer C Role",
    "Polymer D Name", "Polymer D Type", "Polymer D Role",
    "Polymer E Name", "Polymer E Type", "Polymer E Role",
    "Polymer Structure Type",
    "Copolymerized/Blended/Crosslinked/Filled With",
    "Material A Name", "Material A Type", "Material A Role",
    "Material B Name", "Material B Type", "Material B Role",
    "Material C Name", "Material C Type", "Material C Role",
    "Material D Name", "Material D Type", "Material D Role",
    "Material E Name", "Material E Type", "Material E Role",
    "Tensile Strength",
    "Elongation at Break",
    "Young's Modulus",
    "Flexural Modulus",
    "Impact Strength",
    "Stress-Strain",
    "Hardness",
    "Glass Transition",
    "Melting Point"
]

FIELD_ALIASES = {
    "Polymers": [
        "polymer", "copolymer", "blend", "biopolymer",
        "PLA", "polylactic acid", "PCL", "polycaprolactone",
        "PET", "polyethylene terephthalate", "PMMA", "polymethyl methacrylate",
        "PU", "polyurethane", "PA", "polyamide", "nylon",
        "PVC", "polyvinyl chloride", "PVA", "polyvinyl alcohol",
        "PAN", "polyacrylonitrile", "PVP", "polyvinylpyrrolidone",
        "PDMS", "polydimethylsiloxane", "PC", "polycarbonate",
        "PBS", "polybutylene succinate", "PGA", "poly(γ-glutamic acid)",
        "PE", "polyethylene", "PP", "polypropylene", "polyester",
        "GelMA", "gelatin methacrylate", "gelatin", "collagen",
        "chitosan", "alginate", "sodium alginate", "cellulose",
        "nanocellulose", "pectin", "lignin", "starch", "hyaluronic acid",
        "silk fibroin", "polyethylene glycol", "polydopamine",
        "polyacrylamide"
    ],
    "Additives or Modifiers": [
        "additive", "modifier", "plasticizer", "compatibilizer", "filler", "blend",
        "hybrid", "nanocomposite", "composite",
        "nanoparticle", "nanofiller", "nanoclay", "TiO2", "SiO2", "ZnO", "CaCO3",
        "clay", "montmorillonite", "halloysite", "bentonite",
        "PBAT", "PEG", "PHA", "PBSA", "PPC", "EVA", "PLA-g-MA",
        "cellulose nanocrystal", "microcrystalline cellulose", "hemicellulose",
        "soy protein", "wheat bran", "rice husk",
        "CNT", "carbon nanotube", "carbon black", "graphene", "graphene oxide",
        "reduced graphene oxide",
        "fiber", "natural fiber", "glass fiber", "bamboo fiber", "hemp fiber",
        "basalt fiber", "jute fiber", "kenaf fiber",
        "glycerol", "triacetin", "citrate", "ATBC", "TEC", "tributyl citrate",
        "polyethylene glycol",
        "blending", "blended", "copolymerized", "reactive compatibilization",
        "immiscible", "miscible",
        "antioxidant", "nucleating agent", "chain extender", "crosslinker",
        "UV stabilizer", "thermal stabilizer", "fire retardant", "flame retardant"
    ],
    "mechanical properties": [
        "tensile strength", "breaking strength", "tensile properties",
        "elongation at break", "breaking elongation",
        "young's modulus", "tensile modulus",
        "flexural modulus", "bending modulus", "flexural stiffness",
        "impact strength", "impact toughness",
        "stress-strain", "mechanical behavior",
        "hardness", "shore hardness", "rockwell", "durometer",
    ],
    "Glass Transition": ["glass transition", "glass temperature", "Glass Transition Temperature", "Tg"],
    "Melting Point": ["melting point", "melting temperature", "Tm"]
}

# ================= Helper Functions =================

def normalize_field_name(key: str) -> str:
    key = key.strip()
    if key in FIELDS:
        return key
    base = re.sub(r"[（(][^）)]*[）)]", "", key)
    base = re.sub(r"[^\w\s\-/]", "", base)
    low = base.strip().lower()

    # Explicitly handle mechanical property field mapping
    mech_map = {
        "tensile strength": "Tensile Strength",
        "elongation at break": "Elongation at Break",
        "youngs modulus": "Young's Modulus",
        "flexural modulus": "Flexural Modulus",
        "impact strength": "Impact Strength",
        "stressstrain": "Stress-Strain",
        "hardness": "Hardness"
    }
    if low in mech_map:
        return mech_map[low]

    # Handle Tg and Tm mapping
    if low in ["glass transition", "glass transition temperature", "tg"]:
        return "Glass Transition"
    if low in ["melting point", "melting temperature", "tm"]:
        return "Melting Point"

    # Fallback FIELDS correspondence
    for std in FIELDS:
        if low == std.lower():
            return std
    return base

def flatten_material_additives(mat_obj):
    flattened = {}
    if isinstance(mat_obj, dict):
        for mat_key, attrs in mat_obj.items():
            if isinstance(attrs, dict):
                for subk, subv in attrs.items():
                    flattened[f"{mat_key} {subk}"] = subv
            else:
                flattened[mat_key] = attrs
    elif isinstance(mat_obj, list):
        for one in mat_obj:
            flattened.update(flatten_material_additives(one))
    return flattened

def extract_json_blocks(content: str):
    """Extract JSON blocks from text, supporting Markdown format and plain text format"""
    blocks = re.findall(r"```json\s*(.*?)\s*```", content, flags=re.DOTALL)
    if blocks:
        return [b.strip() for b in blocks]
    
    # fallback: Extract content after '🧾 Extraction Result:' (Matches both Chinese and English)
    m = re.search(r"🧾\s*(?:提取结果|Extraction Result)：\s*([\s\S]+)", content)
    if not m:
        return []
    raw = m.group(1).strip()

    if raw.upper().startswith("NONE") or "无响应" in raw or "NO RESPONSE" in raw.upper():
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, (list, dict)):
            return [raw]
    except json.JSONDecodeError:
        pass

    # fallback: Try to match list or dict bracket structure
    start_list = raw.find("[")
    end_list = raw.rfind("]") + 1
    if 0 <= start_list < end_list:
        return [raw[start_list:end_list]]

    start_dict = raw.find("{")
    end_dict = raw.rfind("}") + 1
    if 0 <= start_dict < end_dict:
        return [raw[start_dict:end_dict]]

    return []

def summarize_field_coverage(rows):
    counter = Counter()
    total = len(rows)
    if total == 0:
        print("\n📊 No data rows extracted")
        return

    for row in rows:
        for field in FIELDS:
            if row.get(field, "") not in ["", None, "null"]:
                counter[field] += 1

    print("\n📊 Field Coverage Statistics:")
    for field in FIELDS:
        count = counter.get(field, 0)
        print(f"{field:<45} : {count}/{total} ({count/total:.1%})")

def convert_all_output_folders(root_folder: str, output_csv_path: str):
    print(f"📂 Processing directory: {root_folder}")
    fieldnames = ["Source"] + FIELDS
    all_rows = []

    if not os.path.exists(root_folder):
        print(f"❌ Error: Input directory does not exist {root_folder}")
        return

    for dirpath, _, filenames in os.walk(root_folder):
        # Your pipeline usually generates folders containing 'extracted_output'
        # If some folders are named differently, uncomment the lines below to filter
        # if "extracted_output" not in dirpath.lower():
        #     continue
        
        for fn in filenames:
            if not fn.lower().endswith(".txt"):
                continue
            file_path = os.path.join(dirpath, fn)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            blocks = extract_json_blocks(content)
            if not blocks:
                continue

            for idx, raw_json in enumerate(blocks, start=1):
                try:
                    data = json.loads(raw_json)
                except json.JSONDecodeError:
                    continue
                
                if data is None:
                    continue
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    continue

                for j, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue

                    row = {"Source": f"{fn}_Part{idx}_Material{j+1}"}
                    cleaned = {}

                    # Expand Base Polymer(s)
                    base_polymer_data = item.get("Base Polymer(s)")
                    if isinstance(base_polymer_data, dict):
                        for label, subdata in base_polymer_data.items():
                            if isinstance(subdata, dict):
                                name = subdata.get("Name", "")
                                ptype = subdata.get("Type", "")
                                role = subdata.get("Role", "")
                                cleaned[f"{label} Name"] = name
                                cleaned[f"{label} Type"] = ptype
                                cleaned[f"{label} Role"] = role

                    for k, v in item.items():
                        if k == "Base Polymer(s)":
                            continue

                        nk = normalize_field_name(k)

                        # Handle Other Material
                        if nk.startswith("Other Material") and isinstance(v, dict):
                            mats = flatten_material_additives(v)
                            for mat_field, mat_val in mats.items():
                                cleaned[mat_field] = mat_val
                            continue

                        # Handle Mechanical Properties
                        if nk == "Mechanical Properties" and isinstance(v, dict):
                            for mech_k, mech_v in v.items():
                                if not isinstance(mech_v, dict):
                                    continue
                                val = mech_v.get("value")
                                unit = mech_v.get("unit")
                                if val is None:
                                    cleaned[mech_k] = ""
                                elif isinstance(val, list):
                                    val_str = f"{val[0]}–{val[1]}" if len(val) == 2 else ", ".join(map(str, val))
                                    cleaned[mech_k] = f"{val_str} {unit}".strip() if unit else val_str
                                else:
                                    cleaned[mech_k] = f"{val} {unit}".strip() if unit else str(val)
                            continue

                        # Handle Tg / Tm
                        if nk in ["Glass Transition", "Melting Point"] and isinstance(v, dict):
                            val = v.get("value")
                            unit = v.get("unit")
                            if val is not None:
                                if isinstance(val, list) and len(val) == 2:
                                    val_str = f"{val[0]}–{val[1]}"
                                else:
                                    val_str = str(val)
                                cleaned[nk] = f"{val_str} {unit}".strip() if unit else val_str
                            else:
                                cleaned[nk] = ""
                            continue

                        # Other regular fields
                        if isinstance(v, list):
                            raw_str = ", ".join(str(x) for x in v)
                        elif isinstance(v, dict) and "value" in v and "unit" in v:
                            val = v.get("value")
                            unit = v.get("unit")
                            if val is not None:
                                if isinstance(val, list) and len(val) == 2:
                                    val_str = f"{val[0]}–{val[1]}"
                                else:
                                    val_str = str(val)
                                raw_str = f"{val_str} {unit}".strip() if unit else val_str
                            else:
                                raw_str = ""
                        else:
                            raw_str = "" if v is None else str(v)

                        if re.fullmatch(r"\[\s*(?:\.\.|…)+\s*\]", raw_str):
                            raw_str = ""

                        cleaned[nk] = raw_str

                    # Field alignment padding
                    for col in FIELDS:
                        if col in cleaned:
                            row[col] = cleaned[col]
                        else:
                            found = False
                            for k2 in cleaned:
                                if normalize_field_name(k2) == col:
                                    row[col] = cleaned[k2]
                                    found = True
                                    break
                            if not found:
                                row[col] = ""

                    all_rows.append(row)

    # Write CSV file
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"\n✅ Extraction complete, total {len(all_rows)} records")
        print(f"📁 Result saved to: {output_csv_path}")
        summarize_field_coverage(all_rows)
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

# ================= Main Execution Block =================
# Note: The following part is for compatibility with pipeline.py string replacement logic
# Please do not arbitrarily modify the assignment format of root and out_csv

if __name__ == "__main__":
    # 1. Define placeholder paths (Pipeline will look for and replace these specific strings)
    root = r"Your_split_pdfs_Path"
    out_csv = os.path.join(root, "Extraction_Result.csv")
    
    # 2. Check if running as an independent script (not replaced)
    # If running this script directly for testing, please manually modify the paths below
    if "Your_split_pdfs_Path" in root:
        print("⚠️ Warning: Detected use of default placeholder path.")
        print("If manual testing, please modify the root path in the code.")
        # Example manual test path (uncomment to use):
        # root = r"Data/Data_split"
        # out_csv = r"Data/Processed_Results/Manual_Test.csv"
    
    # 3. Execute conversion
    if os.path.exists(root) and "Your_split_pdfs" not in root:
        convert_all_output_folders(root, out_csv)
    elif "Your_split_pdfs" in root:
        print("❌ Path not configured, cannot run. Please start via pipeline or set path manually.")