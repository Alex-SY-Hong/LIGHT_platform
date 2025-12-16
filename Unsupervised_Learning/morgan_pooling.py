#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen

# ==== RDKit check ====
def _check_rdkit():
    try:
        from rdkit import Chem  # noqa
        from rdkit.Chem import rdFingerprintGenerator  # noqa
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa
            _has_std = True
        except Exception:
            _has_std = False
        return True, "", _has_std
    except Exception as e:
        return False, str(e), False

_RDOK, _RDErr, _HAS_STD = _check_rdkit()
if not _RDOK:
    raise RuntimeError(
        "RDKit not available. Install on Linux via:\n"
        "  conda install -c conda-forge rdkit -y\n"
        f"Original error: {_RDErr}"
    )

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
if _HAS_STD:
    from rdkit.Chem.MolStandardize import rdMolStandardize

DEFAULT_POLYMER_COLS: List[str] = [
    "Polymer A SMILE","Polymer B SMILE","Polymer C SMILE","Polymer D SMILE","Polymer E SMILE"
]
DEFAULT_OTHER_COLS: List[str] = [
    "Material A SMILE","Material B SMILE","Material C SMILE","Material D SMILE","Material E SMILE"
]

# ---------- robust CSV reader (encoding/sep auto) ----------
_DEFAULT_ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "cp936", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]
_DEFAULT_SEPARATORS = [",", "\t", ";", "|"]

def read_csv_robust(path: str,
                    encoding: Optional[str] = None,
                    sep: Optional[str] = None,
                    low_memory: bool = False) -> pd.DataFrame:
    if encoding is not None and sep is not None:
        return pd.read_csv(path, encoding=encoding, sep=sep, low_memory=low_memory)
    enc_list = [encoding] + _DEFAULT_ENCODINGS if encoding else _DEFAULT_ENCODINGS
    sep_list = [sep] + _DEFAULT_SEPARATORS if sep else _DEFAULT_SEPARATORS
    last_err = None
    for enc in enc_list:
        for sp in sep_list:
            try:
                return pd.read_csv(path, encoding=enc, sep=sp, low_memory=low_memory)
            except Exception as e:
                last_err = e
                continue
    # fallback: binary read -> utf-8 ignore -> try common seps
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
        from io import StringIO
        for sp in sep_list:
            try:
                return pd.read_csv(StringIO(raw), sep=sp, low_memory=low_memory)
            except Exception:
                continue
    except Exception:
        pass
    raise last_err if last_err is not None else RuntimeError("Failed to read CSV with multiple encodings/separators")

# ---------- cell cleaning & multi-molecule parsing ----------
_BAD_WS = ["\u200b", "\u200c", "\u200d", "\ufeff"]
_TRANS = {
    "，": ",", "、": ",", "；": ";", "：": ":", "（": "(", "）": ")",
    "【": "[", "】": "]", "“": '"', "”": '"', "‘": "'", "’": "'",
}
_MULTI_DELIMS = re.compile(r"[;|,]+")

def clean_smiles_string(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    for z in _BAD_WS: s = s.replace(z, "")
    for k, v in _TRANS.items(): s = s.replace(k, v)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1].strip()
    if s.lower() in {"na","none","nan","null"}: return ""
    return s

def split_multi_smiles(cell: str) -> List[str]:
    base = clean_smiles_string(cell)
    if not base: return []
    # 如果整个就是可解析 SMILES，就直接返回
    try:
        if Chem.MolFromSmiles(base) is not None:
            return [base]
    except Exception:
        pass
    parts = [p.strip() for p in _MULTI_DELIMS.split(base) if p.strip()]
    out, seen = [], set()
    for p in parts:
        cp = clean_smiles_string(p)
        if cp and cp not in seen:
            seen.add(cp); out.append(cp)
    return out

def is_valid_smiles(s: str) -> bool:
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False

# ---------- SMILES standardization ----------
class _SmilesStandardizer:
    def __init__(self, use_std: bool):
        self.use_std = use_std
        if use_std:
            self.cleaner = rdMolStandardize.CleanupParameters()
            self.largest = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
            self.norm = rdMolStandardize.Normalizer()
            self.reion = rdMolStandardize.Reionizer()
            self.uncharger = rdMolStandardize.Uncharger()
            self.meta_disconnector = rdMolStandardize.MetalDisconnector()

    def standardize_mol(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        if mol is None:
            return None
        try:
            if not self.use_std:
                return mol
            mol = rdMolStandardize.Cleanup(mol, self.cleaner)
            mol = self.meta_disconnector.Disconnect(mol)
            mol = self.norm.normalize(mol)
            mol = self.reion.reionize(mol)
            mol = self.uncharger.uncharge(mol)
            mol = self.largest.choose(mol)
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def canonical_smiles(self, smi: str) -> str:
        if not isinstance(smi, str) or not smi.strip():
            return ""
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = self.standardize_mol(mol)
            if mol is None: return ""
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception:
            return ""

_STD = _SmilesStandardizer(use_std=_HAS_STD)

# ---------- column detection ----------
_SMILES_COL_PAT = re.compile(r"(?:\bsmiles?\b)|(?:\bsmile\b)", re.IGNORECASE)
_RATIO_CAND_KEYS = ["mol%", "wt%", "mass%", "vol%", "ratio", "content", "frac", "loading"]

def autodetect_smiles_cols(df: pd.DataFrame, min_valid_frac: float = 0.2, sample_n: int = 200) -> List[str]:
    cols = []
    name_hits = [c for c in df.columns if _SMILES_COL_PAT.search(c)]
    cols.extend(name_hits)
    check_cols = [c for c in df.columns if c not in cols]
    sub = df[check_cols].head(sample_n)
    for c in check_cols:
        vals = sub[c].dropna().astype(str).tolist()
        if not vals: continue
        ok, checked = 0, 0
        for v in vals:
            toks = split_multi_smiles(v)
            for t in toks[:3]:  # 采样少量 token
                checked += 1
                if is_valid_smiles(t): ok += 1
                if checked >= 50: break
            if checked >= 50: break
        if checked > 0 and (ok / checked) >= min_valid_frac:
            cols.append(c)
    seen, out = set(), []
    for c in df.columns:
        if c in cols and c not in seen:
            seen.add(c); out.append(c)
    return out

def _find_target_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Swelling Ratio (times)"
    ]
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+","", s.lower())
    norm_map = {c: norm(c) for c in df.columns}
    cand_norm = [norm(c) for c in candidates]
    for c, cn in norm_map.items():
        if cn in cand_norm: return c
    for c in df.columns:
        if ("young" in c.lower()) and ("modulus" in c.lower()): return c
    return None

def _find_id_cols(df: pd.DataFrame) -> List[str]:
    keys = ["id","name","样品","配方","recipe","formula","编号"]
    return [c for c in df.columns if any(k in c.lower() for k in keys)]

def _autodetect_ratio_cols(df: pd.DataFrame, smiles_cols: List[str]) -> Optional[List[Optional[str]]]:
    ratio_list: List[Optional[str]] = []
    cols_lower = {c.lower(): c for c in df.columns}
    for col in smiles_cols:
        base = col.replace("SMILES","").replace("SMILE","").strip()
        base_l = base.lower()
        found = None
        for k in _RATIO_CAND_KEYS:
            cand1 = f"{k} {base_l}"
            cand2 = f"{base_l} {k}"
            for ck, orig in cols_lower.items():
                if cand1 in ck or cand2 in ck:
                    found = orig; break
            if found: break
        if not found:
            idx = list(df.columns).index(col)
            window = list(range(max(0, idx-2), min(len(df.columns), idx+3)))
            for j in window:
                cname = df.columns[j]
                if any(k in cname.lower() for k in _RATIO_CAND_KEYS):
                    found = cname; break
        ratio_list.append(found)
    return None if all(v is None for v in ratio_list) else ratio_list

# ---------- Hydrogel-oriented SMARTS patterns ----------
_SMARTS = {
    # Hydrophilic / H-bond
    "OH": "[OX2H]",                        # hydroxyl
    "COOH": "C(=O)[OX1H0-,OX2H1]",         # acid/acetate
    "Ester": "C(=O)O",                      # ester
    "Amide": "C(=O)N",                      # amide
    "Ether": "[OD2]([#6])[#6]",             # R-O-R
    "Urethane": "OC(=O)N",                  # carbamate
    "Carbonate": "OC(=O)O",

    # Ionic / ionizable
    "Amine_primary": "[NX3;H2;!$([NX3](=O)=O)]",
    "Amine_secondary": "[NX3;H1;!$([NX3](=O)=O)]",
    "Amine_tertiary": "[NX3;H0;!$([NX3](=O)=O)]",
    "Quat_Ammonium": "[NX4+]",
    "Sulfonate": "S(=O)(=O)[O-]",
    "Phosphate": "P(=O)(O)(O)O",
    "Imidazole": "n1cc[nH]c1",
    "Guanidinium": "N=C(N)N",

    # Crosslink/Reactive
    "Acrylate_like": "C=CC(=O)O",           # acrylate/methacrylate proxy
    "Epoxide": "C1OC1",                     # 3-membered epoxide
    "Isocyanate": "N=C=O",
    "Maleimide": "O=C1NC(=O)C=CC1=O",

    # Hydrolysable
    "Acetal": "[CX4](-O)(-O)",
    "Carbamate": "N-C(=O)O",

    # General polarity / aromatics
    "Aromatic": "a",
}
from rdkit import Chem as _Chem
_SMARTS_PAT = {k: _Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}

# ---------- Morgan generator & fingerprint ----------
def build_morgan_generator(
    radius: int,
    fp_size: int,
    use_chirality: bool,
    use_features: bool,
    use_bond_types: bool,
    include_ring: bool,
):
    """
    跨版本稳健创建 Morgan 生成器：
    - use_features=True 且可用时，优先用 GetMorganFeatureGenerator
    - 否则用 GetMorganGenerator
    - 兼容是否支持 includeRingMembership / 关键字参数
    """
    funcs = []
    if use_features and hasattr(rdFingerprintGenerator, "GetMorganFeatureGenerator"):
        funcs.append(rdFingerprintGenerator.GetMorganFeatureGenerator)
    funcs.append(rdFingerprintGenerator.GetMorganGenerator)

    last_err = None
    for fn in funcs:
        try:
            return fn(
                radius=radius,
                fpSize=fp_size,
                includeChirality=use_chirality,
                useBondTypes=use_bond_types,
                includeRingMembership=include_ring,
            )
        except TypeError as e:
            last_err = e
        try:
            return fn(
                radius=radius,
                fpSize=fp_size,
                includeChirality=use_chirality,
                useBondTypes=use_bond_types,
            )
        except TypeError as e:
            last_err = e
        try:
            return fn(radius, False, use_chirality, use_bond_types, False, include_ring, None, fp_size)
        except TypeError as e:
            last_err = e
            continue
    raise last_err or RuntimeError("Failed to create Morgan generator (RDKit API mismatch).")


def morgan_from_smi(
    smi: str,
    gen,
    nbits: int = 2048,
    fp_type: str = "count"  # "count" or "bit"
) -> np.ndarray:
    if not isinstance(smi, str) or not smi.strip():
        return np.zeros(nbits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    try:
        if fp_type == "bit":
            bv = gen.GetFingerprint(mol)
            arr = np.zeros((nbits,), dtype=np.float32)
            for idx in list(bv.GetOnBits()):
                if 0 <= idx < nbits:
                    arr[idx] = 1.0
            return arr
        else:
            sv = gen.GetCountFingerprint(mol)
            arr = np.zeros((nbits,), dtype=np.float32)
            for idx, val in sv.GetNonzeroElements().items():
                if 0 <= idx < nbits:
                    arr[idx] = float(val)
            return arr
    except Exception:
        return np.zeros(nbits, dtype=np.float32)

def _calc_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    if mol is None:
        return {}
    return {
        "desc_TPSA": rdMolDescriptors.CalcTPSA(mol),
        "desc_LogP": Crippen.MolLogP(mol),
        "desc_LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "desc_HBA": Lipinski.NumHAcceptors(mol),
        "desc_HBD": Lipinski.NumHDonors(mol),
        "desc_RotB": Lipinski.NumRotatableBonds(mol),
        "desc_FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "desc_RingCount": rdMolDescriptors.CalcNumRings(mol),
        "desc_AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "desc_HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "desc_HeteroAtoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "desc_FormalCharge": float(sum([a.GetFormalCharge() for a in mol.GetAtoms()])),
    }

def _count_smarts(mol: Chem.Mol) -> Dict[str, float]:
    if mol is None:
        return {}
    out = {}
    for name, patt in _SMARTS_PAT.items():
        if patt is not None:
            try:
                out[f"frag_{name}"] = float(len(mol.GetSubstructMatches(patt)))
            except Exception:
                out[f"frag_{name}"] = 0.0
        else:
            out[f"frag_{name}"] = 0.0
    return out

def _calc_hydrogel_indices(d: Dict[str, float]) -> Dict[str, float]:
    # Build simple proxies (avoid div by zero)
    heavy = max(d.get("desc_HeavyAtomCount", 0.0), 1.0)
    hydrophil_index = (d.get("desc_HBA", 0.0) + d.get("desc_HBD", 0.0)) / heavy
    ionic_proxy = sum([
        d.get("frag_Quat_Ammonium", 0.0),
        d.get("frag_Sulfonate", 0.0),
        d.get("frag_Phosphate", 0.0),
        d.get("desc_FormalCharge", 0.0),
    ])
    crosslink_proxy = sum([
        d.get("frag_Acrylate_like", 0.0),
        d.get("frag_Epoxide", 0.0),
        d.get("frag_Isocyanate", 0.0),
        d.get("frag_Maleimide", 0.0),
    ])
    return {
        "idx_Hydrophilicity": hydrophil_index,
        "idx_IonicityProxy": ionic_proxy,
        "idx_CrosslinkProxy": crosslink_proxy,
    }

# ---------- weights ----------
def _build_weights_for_row(
    row: pd.Series,
    smiles_cols: List[str],
    polymer_cols_set: set,
    ratio_cols: Optional[List[Optional[str]]] = None,
    alpha: float = 2.0
) -> np.ndarray:
    w: List[float] = []
    for idx, col in enumerate(smiles_cols):
        smi = row.get(col, None)
        if not isinstance(smi, str) or not smi.strip():
            w.append(0.0); continue
        base = 1.0
        if ratio_cols is not None and idx < len(ratio_cols):
            rname = ratio_cols[idx]
            if rname is not None and rname in row:
                try:
                    val = float(row[rname]); base = 0.0 if np.isnan(val) else float(val)
                except Exception:
                    base = 1.0
        boost = alpha if col in polymer_cols_set else 1.0
        w.append(base * boost)
    w_arr = np.asarray(w, dtype=np.float64)
    if w_arr.sum() <= 0:
        mask = np.array([isinstance(row.get(c,''), str) and row.get(c,'').strip() != '' for c in smiles_cols], dtype=float)
        w_arr = (mask / mask.sum()) if mask.sum() > 0 else mask
    else:
        w_arr = w_arr / w_arr.sum()
    return w_arr.astype(np.float64)

# ---------- main logic ----------
def run_pooling(
    in_csv: str,
    out_csv: Optional[str],
    polymer_cols: List[str],
    other_cols: List[str],
    ratio_cols: Optional[List[str]],
    auto_detect_ratio: bool,
    alpha: float,
    radius: int,
    nbits: int,
    keep_cols_extra: Optional[List[str]],
    target_col_hint: Optional[str],
    encoding: Optional[str],
    sep: Optional[str],
    emit_std_cols: bool = False,
    # new Morgan options
    fp_type: str = "count",
    use_chirality: bool = False,
    use_features: bool = False,
    use_bond_types: bool = True,
    include_ring: bool = False,
    post_binarize: bool = False,
    post_norm: str = "none",
    # multi-molecule
    multi_cell_strategy: str = "avg",
    
    emit_desc: bool = False,
    emit_frags: bool = False,
    emit_indices: bool = False,
    emit_mixstats: bool = False,
    emit_pairwise: bool = False,

) -> pd.DataFrame:
    if not os.path.isfile(in_csv):
        raise FileNotFoundError(f"Input file not found: {in_csv}")

    df = read_csv_robust(in_csv, encoding=encoding, sep=sep, low_memory=False)

    # columns
    requested_cols = [c for c in (list(polymer_cols) + list(other_cols)) if c]
    smiles_cols = [c for c in requested_cols if c in df.columns]
    if not smiles_cols:
        smiles_cols = autodetect_smiles_cols(df, min_valid_frac=0.2, sample_n=200)
        print(f"[INFO] autodetected SMILES columns: {smiles_cols}")
    if not smiles_cols:
        raise ValueError("No SMILES columns found. Check column names or ensure autodetection can find them.")

    polymer_set = set([c for c in polymer_cols if c in smiles_cols])

    # standardization
    std_cache: Dict[str, str] = {}
    def _std_one(s: str) -> str:
        if not isinstance(s, str) or not s.strip():
            return ""
        if s in std_cache:
            return std_cache[s]
        out = _STD.canonical_smiles(s)
        std_cache[s] = out
        return out

    std_smiles_cols: List[str] = []
    if emit_std_cols:
        for col in smiles_cols:
            newc = f"{col} (std)"
            def _std_cell(cell):
                toks = split_multi_smiles(cell) if isinstance(cell, str) else []
                ts = [t for t in ([_std_one(t) for t in toks]) if t]
                return "||".join(ts)
            df[newc] = df[col].map(_std_cell)
            std_smiles_cols.append(newc)
        fp_source_cols = std_smiles_cols
    else:
        fp_source_cols = smiles_cols

    # ratios
    if auto_detect_ratio and (ratio_cols is None or len(ratio_cols) == 0):
        rc_global = _autodetect_ratio_cols(df, smiles_cols)
    else:
        rc_global = list(ratio_cols) if ratio_cols is not None else None
        if rc_global is not None and len(rc_global) != len(smiles_cols):
            rc_global = (rc_global + [None]*(len(smiles_cols)-len(rc_global)))[:len(smiles_cols)]

    # target
    target_col = target_col_hint if (target_col_hint is not None and target_col_hint in df.columns) else _find_target_col(df)

    # Morgan generator
    gen = build_morgan_generator(
        radius=radius,
        fp_size=nbits,
        use_chirality=use_chirality,
        use_features=use_features,
        use_bond_types=use_bond_types,
        include_ring=include_ring,
    )

    # caches
    fp_cache: Dict[Tuple, np.ndarray] = {}
    fail_rows = []

    def _cell_fp_from_tokens(tokens: List[str]) -> np.ndarray:
        vecs = []
        for t in tokens:
            if not t: continue
            key = (t, fp_type, radius, nbits, use_features, use_chirality, use_bond_types, include_ring)
            arr = fp_cache.get(key)
            if arr is None:
                std = _std_one(t)
                if not std: 
                    continue
                arr = morgan_from_smi(std, gen=gen, nbits=nbits, fp_type=fp_type)
                fp_cache[key] = arr
            vecs.append(arr)
        if not vecs:
            return np.zeros(nbits, dtype=np.float32)
        if multi_cell_strategy == "first":
            return vecs[0]
        elif multi_cell_strategy == "sum":
            return np.sum(vecs, axis=0, dtype=np.float32)
        else:  # avg
            return np.mean(vecs, axis=0, dtype=np.float32)
            
    def _lowdim_from_tokens(tokens: List[str]) -> Dict[str, float]:
        # 聚合：对同一格的多个分子，取均值
        desc_sum, frag_sum, idx_sum, n = {}, {}, {}, 0.0
        for t in tokens:
            if not t: 
                continue
            std = _std_one(t)
            mol = Chem.MolFromSmiles(std) if std else None
            if mol is None:
                continue
            d = _calc_descriptors(mol) if emit_desc else {}
            f = _count_smarts(mol) if emit_frags else {}
            di = _calc_hydrogel_indices({**d, **f}) if emit_indices else {}
            def _acc(dst, src):
                for k, v in src.items():
                    dst[k] = dst.get(k, 0.0) + float(v)
            _acc(desc_sum, d); _acc(frag_sum, f); _acc(idx_sum, di); n += 1.0
        def _avg(dct):
            return {k: (v / n) for k, v in dct.items()} if n > 0 else {}
        return {**_avg(desc_sum), **_avg(frag_sum), **_avg(idx_sum)}

    feats: List[np.ndarray] = []
    low_list: List[Dict[str, float]] = []

    for i, row in df.iterrows():
        # 1) 解析每列 SMILES -> tokens
        smi_tokens_per_col: List[List[str]] = []
        for col in fp_source_cols:
            raw = row.get(col, None)
            if not isinstance(raw, str):
                smi_tokens_per_col.append([])
                continue
            if emit_std_cols:
                tokens = [t for t in raw.split("||") if t.strip()]
            else:
                tokens = split_multi_smiles(raw)
            smi_tokens_per_col.append(tokens)

        # 2) 计算配方权重（利用比例列 + polymer 放大因子）
        ws = _build_weights_for_row(row, smiles_cols, polymer_set, ratio_cols=rc_global, alpha=alpha)

        # 3) 计算每列的指纹 & 低维特征
        fp_list = []
        low_per_col: List[Dict[str, float]] = []
        for tokens in smi_tokens_per_col:
            vec = _cell_fp_from_tokens(tokens)
            fp_list.append(vec)
            low_per_col.append(_lowdim_from_tokens(tokens))

        # 4) 记录失败单元格（该格有token但指纹全0）
        for j, tokens in enumerate(smi_tokens_per_col):
            if tokens and np.allclose(fp_list[j], 0.0):
                fail_rows.append({"row": int(i), "column": fp_source_cols[j], "raw": str(row.get(fp_source_cols[j], ""))})

        # 5) 指纹按配方权重加权
        H = np.stack(fp_list, axis=0) if fp_list else np.zeros((1, nbits), dtype=np.float32)
        h_mix = (ws[:, None] * H).sum(axis=0) if len(fp_list) else np.zeros(nbits, dtype=np.float32)

        # 6) 低维特征按配方权重加权（列对齐）
        low_keys = sorted(set().union(*[d.keys() for d in low_per_col])) if low_per_col else []
        low_weighted = {}
        for k in low_keys:
            s = 0.0
            for j, d in enumerate(low_per_col):
                s += ws[j] * float(d.get(k, 0.0))
            low_weighted[k] = float(s)

        # 6.1 可选：配方层面的非加权统计（体现异质性）
        if emit_mixstats and low_per_col:
            for k in low_keys:
                vals = [float(d.get(k, 0.0)) for d in low_per_col if k in d]
                if vals:
                    low_weighted[f"{k}__min"] = float(np.min(vals))
                    low_weighted[f"{k}__max"] = float(np.max(vals))
                    low_weighted[f"{k}__var"] = float(np.var(vals))

        # 6.2 可选：组分两两 Tanimoto 相似度统计（基于 count 向量）
        if emit_pairwise and len(fp_list) >= 2:
            def tanimoto(a, b):
                ab = float(np.minimum(a, b).sum())
                aa = float(a.sum())
                bb = float(b.sum())
                den = aa + bb - ab
                return float(ab / den) if den > 0 else 0.0
            pairs = []
            for a in range(len(fp_list)):
                for b in range(a + 1, len(fp_list)):
                    pairs.append(tanimoto(fp_list[a], fp_list[b]))
            if pairs:
                low_weighted["pair_Tani_mean"] = float(np.mean(pairs))
                low_weighted["pair_Tani_min"]  = float(np.min(pairs))
                low_weighted["pair_Tani_max"]  = float(np.max(pairs))

        # 7) 后处理指纹
        if post_binarize:
            h_mix = (h_mix > 0).astype(np.float32)
        if post_norm == "l2":
            norm = np.linalg.norm(h_mix)
            if norm > 0:
                h_mix = (h_mix / norm).astype(np.float32)
        elif post_norm == "max":
            m = float(np.max(np.abs(h_mix)))
            if m > 0:
                h_mix = (h_mix / m).astype(np.float32)

        feats.append(h_mix)
        low_list.append(low_weighted)

        if (i + 1) % 500 == 0:
            print(f"... processed {i+1} rows")


    X = np.vstack(feats).astype(np.float32)
    fp_cols = [f"fp_{i}" for i in range(nbits)]
    out_df = pd.DataFrame(X, columns=fp_cols)
    # —— 拼接低维特征表 —— 
    if low_list:
        low_df = pd.DataFrame(low_list).fillna(0.0)
        out_df = pd.concat([out_df.reset_index(drop=True), low_df.reset_index(drop=True)], axis=1)
    out_df.insert(0, "row_index", np.arange(len(out_df)))

    if target_col is not None:
        out_df.insert(1, target_col, df[target_col].values)

    keep_auto = _find_id_cols(df)
    keep_cols = list(dict.fromkeys((keep_cols_extra or []) + keep_auto))
    if keep_cols:
        out_df = pd.concat([df[keep_cols].reset_index(drop=True), out_df], axis=1)

    if emit_std_cols:
        out_df = pd.concat([df[std_smiles_cols].reset_index(drop=True), out_df], axis=1)

    if out_csv is None:
        base, _ = os.path.splitext(in_csv)
        out_csv = base + "-pooled-morgan.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] written: {out_csv}; shape: {out_df.shape}")
    if target_col is not None:
        print(f"[INFO] target column preserved as-is: '{target_col}' (no unit conversion)")
    else:
        print("[WARN] target column not found; label not written.")

    if fail_rows:
        log_path = Path(out_csv).with_suffix(".smiles_fail.csv")
        pd.DataFrame(fail_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"[WARN] some cells failed to parse; logged to: {log_path}")

    return out_df

def parse_args():
    p = argparse.ArgumentParser(description="Multi-SMILES -> (standardize) -> Morgan -> weighted pooling")
    p.add_argument("--in_csv",  required=True, help="Input CSV containing multiple SMILES columns")
    p.add_argument("--out_csv", default=None, help="Output CSV (default: alongside input as *-pooled-morgan.csv)")
    p.add_argument("--alpha",   type=float, default=3.0, help="Polymer weight amplification factor")
    p.add_argument("--radius",  type=int,   default=2,   help="Morgan radius")
    p.add_argument("--nbits",   type=int,   default=2048,help="Fingerprint length")
    p.add_argument("--no_auto_ratio", action="store_true", help="Disable auto-detection of ratio columns")
    p.add_argument("--ratio_cols", nargs="*", default=None, help="Manually specify ratio columns aligned to SMILES columns")
    p.add_argument("--polymer_cols", nargs="*", default=DEFAULT_POLYMER_COLS, help="Polymer SMILES column names")
    p.add_argument("--other_cols",   nargs="*", default=DEFAULT_OTHER_COLS,   help="Other (filler/additive) SMILES column names")
    p.add_argument("--keep_cols",    nargs="*", default=None, help="Extra identifier columns to keep (e.g., SampleID, Name)")
    p.add_argument("--target_col",   default=None, help="Explicit Young's Modulus column name (if not auto-detected)")
    # encoding / separator
    p.add_argument("--encoding",     default=None, help="CSV encoding (e.g., gbk, utf-8, latin1). If not set, try multiple encodings automatically.")
    p.add_argument("--sep",          default=None, help="CSV separator (e.g., ',', '\\t', ';', '|'). If not set, try common separators automatically.")
    # std / multi-molecule
    p.add_argument("--emit_std_cols", action="store_true", help="Also write standardized SMILES columns with '(std)' suffix (multi-molecule joined by '||')")
    p.add_argument("--multi_cell_strategy", choices=["first","avg","sum"], default="avg",
                   help="If one cell contains multiple molecules, how to aggregate: first/avg/sum (default: avg)")
    # Morgan options
    p.add_argument("--fp_type", choices=["count","bit"], default="count", help="Fingerprint type")
    p.add_argument("--use_chirality", action="store_true", help="Include chirality")
    p.add_argument("--use_features", action="store_true",  help="Morgan feature fingerprint")
    p.add_argument("--no_bond_types", action="store_true", help="Exclude bond types")
    p.add_argument("--include_ring", action="store_true",  help="Include ring membership")
    p.add_argument("--post_binarize", action="store_true", help="Binarize after pooling (>0 -> 1)")
    p.add_argument("--post_norm", choices=["none","l2","max"], default="none", help="Normalize after pooling")
        # —— Hydrogel feature switches ——
    p.add_argument("--emit_desc", action="store_true", help="输出 RDKit 分子描述符（比例加权）")
    p.add_argument("--emit_frags", action="store_true", help="输出水凝胶相关 SMARTS 片段计数（比例加权）")
    p.add_argument("--emit_indices", action="store_true", help="输出水凝胶衍生指数（比例加权）")
    p.add_argument("--emit_mixstats", action="store_true", help="输出配方层面的 min/max/var 统计（按列）")
    p.add_argument("--emit_pairwise", action="store_true", help="输出组分两两 Tanimoto 相似度统计（mean/min/max）")

    return p.parse_args()
    

def main():
    args = parse_args()
    run_pooling(
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        polymer_cols=args.polymer_cols,
        other_cols=args.other_cols,
        ratio_cols=args.ratio_cols,
        auto_detect_ratio=not args.no_auto_ratio,
        alpha=args.alpha,
        radius=args.radius,
        nbits=args.nbits,
        keep_cols_extra=args.keep_cols,
        target_col_hint=args.target_col,
        encoding=args.encoding,
        sep=args.sep,
        emit_std_cols=args.emit_std_cols,
        # Morgan options
        fp_type=args.fp_type,
        use_chirality=args.use_chirality,
        use_features=args.use_features,
        use_bond_types=not args.no_bond_types,
        include_ring=args.include_ring,
        post_binarize=args.post_binarize,
        post_norm=args.post_norm,
        # multi-molecule
        multi_cell_strategy=args.multi_cell_strategy,
        emit_desc=args.emit_desc,
        emit_frags=args.emit_frags,
        emit_indices=args.emit_indices,
        emit_mixstats=args.emit_mixstats,
        emit_pairwise=args.emit_pairwise,

    )

if __name__ == "__main__":
    main()
