# =============================================================================
# NeuroLNP: app.py — Streamlit Web Application
# Hybrid AI = ML (Stacking Ensemble) + Expert Rules Layer (Biophysics)
#
# วิธีรัน:
#   streamlit run app.py
#
# ต้องมีไฟล์ใน directory เดียวกัน:
#   neurolnp_best_model.pkl
#   neurolnp_benchmark_results.json
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json, pickle, os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="NeuroLNP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS — Dark scientific aesthetic
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ─────────────────────────────────────────────────────────────── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --border:    #1e2d40;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --success:   #10b981;
    --warn:      #f59e0b;
    --danger:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --sans:      'Syne', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1400px !important; }

/* ── Typography ───────────────────────────────────────────────────────── */
h1, h2, h3 { font-family: var(--sans) !important; font-weight: 800 !important; }

/* ── Hero header ──────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 50%, #1a0a2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(124,58,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: var(--sans) !important;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: var(--mono) !important;
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    font-family: var(--mono) !important;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-top: 1rem;
    margin-right: 0.5rem;
    letter-spacing: 0.05em;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-success {
    background: rgba(16,185,129,0.05);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--success);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-warn {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid var(--warn);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-danger {
    background: rgba(239,68,68,0.05);
    border: 1px solid rgba(239,68,68,0.2);
    border-left: 3px solid var(--danger);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Score display ────────────────────────────────────────────────────── */
.score-big {
    font-family: var(--mono) !important;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
}
.score-label {
    font-family: var(--mono) !important;
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.grade-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 3rem;
    height: 3rem;
    border-radius: 50%;
    font-family: var(--mono) !important;
    font-size: 1.4rem;
    font-weight: 700;
}

/* ── Rule score bar ───────────────────────────────────────────────────── */
.rule-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.rule-name {
    font-family: var(--mono) !important;
    font-size: 0.72rem;
    color: var(--muted);
    width: 140px;
    flex-shrink: 0;
}
.rule-bar-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 999px;
    overflow: hidden;
}
.rule-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.rule-val {
    font-family: var(--mono) !important;
    font-size: 0.72rem;
    color: var(--text);
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Tags ─────────────────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    font-family: var(--mono) !important;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin: 0.2rem 0.1rem;
}
.tag-good  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.tag-warn  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.tag-bad   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.tag-info  { background: rgba(0,212,255,0.1);   color: #00d4ff; border: 1px solid rgba(0,212,255,0.2); }

/* ── Metric box ───────────────────────────────────────────────────────── */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: var(--mono) !important;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-lbl {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Streamlit overrides ──────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div { color: var(--text) !important; }
.stSlider [data-baseweb="slider"] div { background: var(--accent) !important; }
label, .stCheckbox label { color: var(--text) !important; font-size: 0.85rem !important; }
.stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
}
.stButton button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stExpander {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.stExpander summary { color: var(--text) !important; }
.stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
.stSidebar [data-testid="stSidebarContent"] { background: var(--surface) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Helper Functions
# =============================================================================

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=32)


def get_molecular_features(smiles: str) -> dict | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        heavy = Descriptors.HeavyAtomCount(mol)
        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        tpsa  = Descriptors.TPSA(mol)
        hbd      = Descriptors.NumHDonors(mol)
        hba      = Descriptors.NumHAcceptors(mol)
        rotb     = Descriptors.NumRotatableBonds(mol)
        fsp3     = Descriptors.FractionCSP3(mol)
        nhoh     = Descriptors.NHOHCount(mol)
        no_count = Descriptors.NOCount(mol)
        arom_rings  = rdMolDescriptors.CalcNumAromaticRings(mol)
        ali_rings   = rdMolDescriptors.CalcNumAliphaticRings(mol)
        ring_count  = rdMolDescriptors.CalcNumRings(mol)
        arom_atoms  = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        arom_prop   = arom_atoms / heavy if heavy > 0 else 0
        formal_charge = Chem.GetFormalCharge(mol)
        bertz_ct      = GraphDescriptors.BertzCT(mol)
        chi0v         = GraphDescriptors.Chi0v(mol)
        stereo_cnt    = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        hac_mw        = mw / heavy if heavy > 0 else 0
        logp_tpsa = logp - (tpsa / 100.0)
        egan_ok   = int(logp <= 3.3 and tpsa <= 90)
        n_atoms   = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        o_atoms   = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        s_atoms   = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
        hal_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9,17,35,53])
        fp_obj   = _MORGAN_GEN.GetFingerprint(mol)
        fp_array = np.zeros(32, dtype=np.int8)
        ConvertToNumpyArray(fp_obj, fp_array)
        morgan_feats = {f'MorganFP_{i}': int(v) for i, v in enumerate(fp_array)}
        return {
            'MolWt': mw, 'LogP': logp, 'TPSA': tpsa,
            'NumHDonors': hbd, 'NumHAcceptors': hba,
            'NumRotatableBonds': rotb, 'FractionCSP3': fsp3,
            'HeavyAtomCount': heavy, 'NHOHCount': nhoh, 'NOCount': no_count,
            'NumAromaticRings': arom_rings, 'NumAliphaticRings': ali_rings,
            'NumRings': ring_count, 'AromaticProportion': arom_prop,
            'FormalCharge': formal_charge, 'BertzCT': bertz_ct, 'Chi0v': chi0v,
            'NumStereocenters': stereo_cnt, 'MWperHeavyAtom': hac_mw,
            'LogP_TPSA_ratio': logp_tpsa, 'EganBBB': egan_ok,
            'NumN': n_atoms, 'NumO': o_atoms, 'NumS': s_atoms,
            'NumHalogens': hal_atoms, **morgan_feats,
        }
    except Exception:
        return None


RULE_WEIGHTS = {
    'rule1_tail':     0.25,
    'rule2a_peg_mw':  0.10,
    'rule2b_peg_pct': 0.10,
    'rule3a_size':    0.15,
    'rule3b_zeta':    0.15,
    'rule3c_pdi':     0.10,
    'rule3d_pka':     0.10,
    'rule4_ligand':   0.05,
}


def apply_expert_rules(
    carbon_tail_length, peg_mw, peg_mol_percent,
    particle_size_nm, zeta_potential_mv, pdi, pka,
    has_acetylcholine, targeting_ligand,
) -> dict:
    scores, warnings, recs = {}, [], []

    # Rule 1: Carbon Tail
    tail_map = {12: 0.5, 14: 1.0, 16: 0.6, 18: 0.2}
    scores['rule1_tail'] = tail_map.get(carbon_tail_length, 0.3)
    if carbon_tail_length == 14:
        recs.append("✅ C14: optimal — ลด liver tropism")
    elif carbon_tail_length == 18:
        warnings.append("❌ C18: Liver tropism สูง — ApoE จะชักนำ LNP ไปตับ")
        recs.append("🔧 เปลี่ยนเป็น C14 ทันที")
    elif carbon_tail_length == 16:
        warnings.append("⚠️ C16: เริ่มสะสมตับมากกว่า C14")
        recs.append("🔧 แนะนำเปลี่ยนเป็น C14")
    else:
        warnings.append(f"⚠️ C{carbon_tail_length}: โครงสร้างไม่เสถียรพอ")
        recs.append("🔧 แนะนำเปลี่ยนเป็น C14")

    # Rule 2A: PEG MW
    peg_mw_map = {1000: 0.4, 2000: 1.0, 3000: 0.3, 5000: 0.1}
    scores['rule2a_peg_mw'] = peg_mw_map.get(peg_mw, 0.5)
    if peg_mw == 2000:
        recs.append("✅ PEG-2000: มาตรฐาน — endosomal escape ดี")
    elif peg_mw == 3000:
        warnings.append("❌ PEG-3000: ชั้นหนาเกิน ขวาง endosomal escape")
        recs.append("🔧 ลดเป็น PEG-2000")
    elif peg_mw >= 5000:
        warnings.append(f"❌ PEG-{peg_mw}: หนามากเกิน mRNA ออกไม่ได้")
        recs.append("🔧 เปลี่ยนเป็น PEG-2000 ทันที")
    else:
        warnings.append(f"⚠️ PEG-{peg_mw}: นอก standard range")

    # Rule 2B: PEG mol%
    if 1.5 <= peg_mol_percent <= 2.5:
        scores['rule2b_peg_pct'] = 1.0
        recs.append(f"✅ PEG {peg_mol_percent:.1f} mol%: golden zone")
    elif peg_mol_percent < 1.5:
        scores['rule2b_peg_pct'] = 0.4
        warnings.append(f"⚠️ PEG {peg_mol_percent:.1f} mol%: น้อยเกิน")
        recs.append("🔧 เพิ่ม PEG เป็น 1.5–2.5 mol%")
    elif peg_mol_percent <= 3.5:
        scores['rule2b_peg_pct'] = 0.6
        warnings.append(f"⚠️ PEG {peg_mol_percent:.1f} mol%: มากเกินนิดหน่อย")
        recs.append("🔧 ลดเป็น 1.5–2.5 mol%")
    else:
        scores['rule2b_peg_pct'] = 0.1
        warnings.append(f"❌ PEG {peg_mol_percent:.1f} mol%: มากเกิน")
        recs.append("🔧 ลดเป็น 1.5–2.5 mol% ทันที")

    # Rule 3A: Size
    if 50 <= particle_size_nm <= 150:
        scores['rule3a_size'] = 1.0
        recs.append(f"✅ Size {particle_size_nm:.0f} nm: optimal BBB crossing")
    elif particle_size_nm < 50:
        scores['rule3a_size'] = 0.5
        warnings.append(f"⚠️ Size {particle_size_nm:.0f} nm: เล็กเกิน")
        recs.append("🔧 เพิ่มเป็น 80–120 nm")
    elif particle_size_nm <= 200:
        scores['rule3a_size'] = 0.6
        warnings.append(f"⚠️ Size {particle_size_nm:.0f} nm: ใหญ่เกินนิดหน่อย")
        recs.append("🔧 ลดเป็น < 150 nm")
    else:
        scores['rule3a_size'] = 0.1
        warnings.append(f"❌ Size {particle_size_nm:.0f} nm: ผ่าน BBB ไม่ได้")
        recs.append("🔧 ปรับ formulation ให้ size < 150 nm")

    # Rule 3B: Zeta
    if -10 <= zeta_potential_mv <= -2:
        scores['rule3b_zeta'] = 1.0
        recs.append(f"✅ Zeta {zeta_potential_mv:.1f} mV: stealth mode")
    elif -2 < zeta_potential_mv <= 5:
        scores['rule3b_zeta'] = 0.7
        recs.append(f"✅ Zeta {zeta_potential_mv:.1f} mV: neutral — ยอมรับได้")
    elif 5 < zeta_potential_mv <= 15:
        scores['rule3b_zeta'] = 0.3
        warnings.append(f"⚠️ Zeta +{zeta_potential_mv:.1f} mV: cationic เกิน")
        recs.append("🔧 ลด cationic lipid")
    elif zeta_potential_mv > 15:
        scores['rule3b_zeta'] = 0.0
        warnings.append(f"❌ Zeta +{zeta_potential_mv:.1f} mV: อันตราย ทำลาย BBB")
        recs.append("🔧 ต้องปรับ lipid composition ใหม่ทั้งหมด")
    elif -20 <= zeta_potential_mv < -10:
        scores['rule3b_zeta'] = 0.7
        warnings.append(f"⚠️ Zeta {zeta_potential_mv:.1f} mV: negative เกินนิดหน่อย")
    else:
        scores['rule3b_zeta'] = 0.2
        warnings.append(f"❌ Zeta {zeta_potential_mv:.1f} mV: negative มากเกิน")
        recs.append("🔧 ปรับเป็น -10 ถึง -2 mV")

    # Rule 3C: PDI
    if pdi <= 0.1:
        scores['rule3c_pdi'] = 1.0
        recs.append(f"✅ PDI {pdi:.2f}: monodisperse ดีมาก")
    elif pdi <= 0.2:
        scores['rule3c_pdi'] = 0.8
        recs.append(f"✅ PDI {pdi:.2f}: ยอมรับได้")
    elif pdi <= 0.3:
        scores['rule3c_pdi'] = 0.5
        warnings.append(f"⚠️ PDI {pdi:.2f}: polydisperse เกิน")
        recs.append("🔧 optimize mixing conditions")
    else:
        scores['rule3c_pdi'] = 0.1
        warnings.append(f"❌ PDI {pdi:.2f}: polydisperse มาก")
        recs.append("🔧 ปรับ formulation process ใหม่")

    # Rule 3D: pKa
    if 6.0 <= pka <= 7.0:
        scores['rule3d_pka'] = 1.0
        recs.append(f"✅ pKa {pka:.1f}: optimal endosomal escape")
    elif 5.5 <= pka < 6.0 or 7.0 < pka <= 7.5:
        scores['rule3d_pka'] = 0.6
        warnings.append(f"⚠️ pKa {pka:.1f}: นอก optimal range เล็กน้อย")
        recs.append("🔧 แนะนำ pKa 6.0–7.0")
    else:
        scores['rule3d_pka'] = 0.1
        warnings.append(f"❌ pKa {pka:.1f}: นอก acceptable range (5.5–7.5)")
        recs.append("🔧 ปรับโครงสร้าง ionizable lipid")

    # Rule 4: Ligand
    ligand_map = {
        'acetylcholine': 1.0, 'nicotine': 0.7,
        'glucose': 0.5, 'tryptophan': 0.5,
        'memantine': 0.4, 'none': 0.2,
    }
    lig_key = targeting_ligand.lower().strip()
    scores['rule4_ligand'] = ligand_map.get(lig_key, 0.3)
    if has_acetylcholine and lig_key != 'acetylcholine':
        scores['rule4_ligand'] = max(scores['rule4_ligand'], 0.8)
        recs.append("✅ Acetylcholine conjugation — brain tropism เพิ่ม")
    elif lig_key == 'acetylcholine' or has_acetylcholine:
        recs.append("✅ Acetylcholine: ดีที่สุด → neurons + astrocytes")
    elif lig_key == 'none':
        warnings.append("⚠️ ไม่มี ligand — brain delivery < 1% injected dose")
        recs.append("🔧 เพิ่ม Acetylcholine-PEG-lipid conjugate")
    else:
        recs.append(f"⚠️ {targeting_ligand}: ใช้ได้แต่ไม่ specific เท่า Acetylcholine")

    expert_score_10 = round(
        sum(scores[k] * RULE_WEIGHTS[k] for k in RULE_WEIGHTS) * 10, 2
    )

    if expert_score_10 >= 8:
        tier, overall = 'A', '🟢 EXCELLENT'
    elif expert_score_10 >= 6:
        tier, overall = 'B', '🟡 GOOD'
    elif expert_score_10 >= 4:
        tier, overall = 'C', '🟠 FAIR'
    else:
        tier, overall = 'D', '🔴 POOR'

    return {
        'rule_scores': scores, 'expert_score_10': expert_score_10,
        'tier': tier, 'overall': overall,
        'warnings': warnings, 'recommendations': recs,
    }


@st.cache_resource
def load_model():
    import os
    # หาไฟล์จาก path ของ app2.py เสมอ ไม่ว่าจะรันที่ไหน
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'neurolnp_best_model.pkl')
    meta_path  = os.path.join(BASE_DIR, 'models', 'neurolnp_benchmark_results.json')

    if not os.path.exists(model_path):
        st.error(f"❌ ไม่พบ model ที่: {model_path}")
        return None, None
    if not os.path.exists(meta_path):
        st.error(f"❌ ไม่พบ metadata ที่: {meta_path}")
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    return model, meta


def predict_lnp(smiles, carbon_tail_length, peg_mw, peg_mol_percent,
                particle_size_nm, zeta_potential_mv, pdi, pka,
                has_acetylcholine, targeting_ligand, model, meta) -> dict:

    feats = get_molecular_features(smiles)
    if feats is None:
        return {'error': 'SMILES ไม่ถูกต้อง กรุณาตรวจสอบอีกครั้ง'}

    feat_vector = pd.DataFrame([feats])[meta['feature_names']]
    logbb_pred  = float(model.predict(feat_vector)[0])

    if logbb_pred > 0.3:
        bbb_class, bbb_color = 'BBB+', '#10b981'
        bbb_note = 'ผ่าน BBB ได้ดี'
    elif logbb_pred > -1.0:
        bbb_class, bbb_color = 'BBB±', '#f59e0b'
        bbb_note = 'ผ่านได้ปานกลาง'
    else:
        bbb_class, bbb_color = 'BBB−', '#ef4444'
        bbb_note = 'ผ่าน BBB ไม่ได้'

    ml_score_10 = round(max(0.0, min(10.0,
        (logbb_pred - (-2.69)) / (1.70 - (-2.69)) * 10
    )), 2)

    rules = apply_expert_rules(
        carbon_tail_length, peg_mw, peg_mol_percent,
        particle_size_nm, zeta_potential_mv, pdi, pka,
        has_acetylcholine, targeting_ligand,
    )

    combined = round(0.4 * ml_score_10 + 0.6 * rules['expert_score_10'], 2)
    grade    = 'A' if combined >= 7 else 'B' if combined >= 5 else 'C' if combined >= 3 else 'D'
    grade_color = {'A': '#10b981', 'B': '#00d4ff', 'C': '#f59e0b', 'D': '#ef4444'}[grade]

    return {
        'logBB': logbb_pred, 'bbb_class': bbb_class,
        'bbb_color': bbb_color, 'bbb_note': bbb_note,
        'ml_score': ml_score_10,
        'expert_score': rules['expert_score_10'],
        'expert_tier': rules['tier'],
        'expert_overall': rules['overall'],
        'rule_scores': rules['rule_scores'],
        'warnings': rules['warnings'],
        'recommendations': rules['recommendations'],
        'combined': combined, 'grade': grade, 'grade_color': grade_color,
        'feats': feats,
    }


def score_color(v: float) -> str:
    if v >= 0.8: return '#10b981'
    if v >= 0.6: return '#00d4ff'
    if v >= 0.4: return '#f59e0b'
    return '#ef4444'


def rule_bar_html(name, score, weight) -> str:
    pct   = int(score * 100)
    color = score_color(score)
    return f"""
<div class="rule-row">
  <div class="rule-name">{name}</div>
  <div class="rule-bar-track">
    <div class="rule-bar-fill" style="width:{pct}%; background:{color};"></div>
  </div>
  <div class="rule-val">{score:.2f}</div>
  <div style="font-family:var(--mono);font-size:0.65rem;color:var(--muted);width:40px;">×{weight}</div>
</div>"""


# =============================================================================
# SIDEBAR — LNP Parameters
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
         color:#00d4ff;margin-bottom:1rem;letter-spacing:-0.01em;">
        🔬 LNP Parameters
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Rule 1 — Lipid Tail**")
    carbon_tail = st.selectbox(
        "Carbon tail length",
        options=[12, 14, 16, 18],
        index=1,
        format_func=lambda x: f"C{x} {'⭐ Recommended' if x==14 else ''}",
    )

    st.markdown("---")
    st.markdown("**Rule 2 — PEG-Lipid**")
    peg_mw = st.selectbox(
        "PEG molecular weight (Da)",
        options=[1000, 2000, 3000, 5000],
        index=1,
        format_func=lambda x: f"PEG-{x} {'⭐ Standard' if x==2000 else ''}",
    )
    peg_pct = st.slider("PEG mol%", 0.5, 5.0, 2.0, 0.1,
                         help="Golden zone: 1.5–2.5 mol%")

    st.markdown("---")
    st.markdown("**Rule 3 — Particle Properties**")
    size    = st.slider("Particle size (nm)", 20.0, 300.0, 100.0, 5.0,
                         help="Optimal: 50–150 nm")
    zeta    = st.slider("Zeta potential (mV)", -30.0, 30.0, -5.0, 0.5,
                         help="Optimal: -10 to -2 mV")
    pdi_val = st.slider("PDI", 0.05, 0.5, 0.15, 0.01,
                         help="Optimal: < 0.2")
    pka_val = st.slider("pKa", 4.0, 9.0, 6.5, 0.1,
                         help="Optimal: 6.0–7.0")

    st.markdown("---")
    st.markdown("**Rule 4 — Brain Targeting**")
    ligand  = st.selectbox(
        "Targeting ligand",
        options=['acetylcholine', 'nicotine', 'glucose', 'tryptophan', 'memantine', 'none'],
        format_func=lambda x: f"{x} {'⭐' if x=='acetylcholine' else ''}",
    )
    has_ach = st.checkbox("มี Acetylcholine conjugation?",
                           value=(ligand == 'acetylcholine'))

    st.markdown("---")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
         color:#64748b;line-height:1.8;">
        Model: Stacking Ensemble<br>
        Dataset: B3DB (n=1,058)<br>
        Test R²: 0.4783<br>
        Expert Rules: v1.1 (multiplicative)<br>
        Combined: ML 40% + Expert 60%
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">NeuroLNP</div>
    <div style="font-family:'Space Mono',monospace;font-size:1rem;
         color:#94a3b8;margin-bottom:1rem;">
        Hybrid AI Platform for Brain-Targeted LNP Design
    </div>
    <span class="hero-badge">Stacking Ensemble ML</span>
    <span class="hero-badge">4-Rule Biophysics Engine</span>
    <span class="hero-badge">AChR Targeting</span>
    <span class="hero-badge">B3DB Dataset</span>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model, meta = load_model()

if model is None:
    st.error("""
    ❌ ไม่พบไฟล์ model — กรุณาตรวจสอบว่ามีไฟล์เหล่านี้ใน directory เดียวกัน:
    - `neurolnp_best_model.pkl`
    - `neurolnp_benchmark_results.json`

    รัน `train_model.py` ใน Google Colab ก่อน แล้ว download ไฟล์มาวางใน folder เดียวกับ app.py
    """)
    st.stop()

# ── SMILES Input ──────────────────────────────────────────────────────────────
col_input, col_examples = st.columns([3, 1])

with col_input:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;
         color:#64748b;margin-bottom:0.4rem;text-transform:uppercase;
         letter-spacing:0.08em;">
        SMILES Input — Lipid Component
    </div>
    """, unsafe_allow_html=True)
    smiles_input = st.text_area(
        "SMILES",
        value="Cn1c(=O)c2c(ncn2C)n(c1=O)C",
        height=80,
        label_visibility="collapsed",
        placeholder="ใส่ SMILES ของ lipid component ที่ต้องการทดสอบ...",
    )

with col_examples:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:0.8rem;
         color:#64748b;margin-bottom:0.4rem;text-transform:uppercase;
         letter-spacing:0.08em;">
        ตัวอย่าง
    </div>
    """, unsafe_allow_html=True)
    example = st.selectbox(
        "เลือกสาร",
        options=[
            "— เลือก —",
            "Caffeine",
            "Diazepam",
            "Aspirin",
            "Ibuprofen",
        ],
        label_visibility="collapsed",
    )
    example_smiles = {
        "Caffeine":  "Cn1c(=O)c2c(ncn2C)n(c1=O)C",
        "Diazepam":  "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "Aspirin":   "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    }
    if example != "— เลือก —":
        smiles_input = example_smiles[example]

run = st.button("⚡  ANALYZE LNP")

# ── Results ───────────────────────────────────────────────────────────────────
if run and smiles_input.strip():
    with st.spinner("กำลังวิเคราะห์..."):
        res = predict_lnp(
            smiles_input.strip(),
            carbon_tail, peg_mw, peg_pct,
            size, zeta, pdi_val, pka_val,
            has_ach, ligand, model, meta,
        )

    if 'error' in res:
        st.error(f"❌ {res['error']}")
    else:
        st.markdown("---")

        # ── Row 1: 3 big scores ───────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)

        with c1:
            color = res['bbb_color']
            st.markdown(f"""
            <div class="card" style="text-align:center;border-color:{color}30;
                 background:rgba({
                    '16,185,129' if color=='#10b981' else
                    '245,158,11' if color=='#f59e0b' else
                    '239,68,68'
                 },0.05);">
                <div class="score-label">logBB Predicted</div>
                <div class="score-big" style="color:{color};margin:0.5rem 0;">
                    {res['logBB']:+.3f}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:1rem;
                     font-weight:700;color:{color};">
                    {res['bbb_class']}
                </div>
                <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">
                    {res['bbb_note']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            ec = res['grade_color']
            st.markdown(f"""
            <div class="card" style="text-align:center;border-color:{ec}30;
                 background:rgba(0,212,255,0.03);">
                <div class="score-label">Combined Score</div>
                <div class="score-big" style="color:{ec};margin:0.5rem 0;">
                    {res['combined']:.1f}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                     color:#64748b;">/ 10.0</div>
                <div style="margin-top:0.5rem;">
                    <span class="grade-badge" style="background:{ec}20;
                          border:2px solid {ec};color:{ec};">
                        {res['grade']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="score-label">ML vs Expert</div>
                <div style="display:flex;justify-content:center;gap:1rem;
                     margin:0.8rem 0;align-items:flex-end;">
                    <div>
                        <div style="font-family:'Space Mono',monospace;
                             font-size:2rem;font-weight:700;color:#7c3aed;">
                            {res['ml_score']:.1f}
                        </div>
                        <div class="score-label">ML Score</div>
                    </div>
                    <div style="font-size:1.5rem;color:#1e2d40;padding-bottom:0.5rem;">+</div>
                    <div>
                        <div style="font-family:'Space Mono',monospace;
                             font-size:2rem;font-weight:700;color:#00d4ff;">
                            {res['expert_score']:.1f}
                        </div>
                        <div class="score-label">Expert Score</div>
                    </div>
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                     color:#64748b;">40% ML + 60% Expert Rules</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # ── Row 2: Rule breakdown + Warnings ──────────────────────────────────
        col_rules, col_warn = st.columns([1, 1])

        with col_rules:
            rule_labels = {
                'rule1_tail':     f'C-Tail (C{carbon_tail})',
                'rule2a_peg_mw':  f'PEG-{peg_mw} MW',
                'rule2b_peg_pct': f'PEG {peg_pct:.1f} mol%',
                'rule3a_size':    f'Size {size:.0f} nm',
                'rule3b_zeta':    f'Zeta {zeta:.1f} mV',
                'rule3c_pdi':     f'PDI {pdi_val:.2f}',
                'rule3d_pka':     f'pKa {pka_val:.1f}',
                'rule4_ligand':   f'Ligand ({ligand})',
            }
            bars_html = "".join(
                rule_bar_html(rule_labels[k], res['rule_scores'][k], RULE_WEIGHTS[k])
                for k in RULE_WEIGHTS
            )
            st.markdown(f"""
            <div class="card card-accent">
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                     font-size:0.9rem;margin-bottom:1rem;color:#e2e8f0;">
                    Expert Rules Breakdown
                </div>
                {bars_html}
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                     color:#64748b;margin-top:0.8rem;border-top:1px solid #1e2d40;
                     padding-top:0.6rem;">
                    Expert Score = Σ(rule_score × weight) × 10
                    = {res['expert_score']:.2f}/10
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_warn:
            # Warnings
            if res['warnings']:
                warn_items = "".join(
                    f'<div style="font-size:0.8rem;padding:0.3rem 0;'
                    f'border-bottom:1px solid #1e2d40;color:#e2e8f0;">{w}</div>'
                    for w in res['warnings']
                )
                card_cls = "card-danger" if res['grade'] == 'D' else "card-warn"
                st.markdown(f"""
                <div class="{card_cls}">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         font-size:0.9rem;margin-bottom:0.8rem;">
                        ⚠️ Warnings ({len(res['warnings'])})
                    </div>
                    {warn_items}
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            fixes = [r for r in res['recommendations'] if r.startswith('🔧')]
            goods = [r for r in res['recommendations'] if r.startswith('✅')]
            if goods:
                good_items = "".join(
                    f'<div style="font-size:0.78rem;padding:0.25rem 0;color:#e2e8f0;">{g}</div>'
                    for g in goods
                )
                st.markdown(f"""
                <div class="card-success">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         font-size:0.9rem;margin-bottom:0.6rem;">
                        ✅ Passed ({len(goods)})
                    </div>
                    {good_items}
                </div>
                """, unsafe_allow_html=True)
            if fixes:
                fix_items = "".join(
                    f'<div style="font-size:0.78rem;padding:0.25rem 0;color:#e2e8f0;">{f}</div>'
                    for f in fixes
                )
                st.markdown(f"""
                <div class="card-warn">
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         font-size:0.9rem;margin-bottom:0.6rem;">
                        🔧 Recommendations
                    </div>
                    {fix_items}
                </div>
                """, unsafe_allow_html=True)

        # ── Row 3: Molecular features ──────────────────────────────────────────
        with st.expander("🧪 Molecular Features (ML Input)", expanded=False):
            feats = res['feats']
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            for col_st, label, key in [
                (m1, "Mol Weight", 'MolWt'),
                (m2, "LogP", 'LogP'),
                (m3, "TPSA (Å²)", 'TPSA'),
                (m4, "H-Bond Donors", 'NumHDonors'),
                (m5, "Rot. Bonds", 'NumRotatableBonds'),
                (m6, "Formal Charge", 'FormalCharge'),
            ]:
                val = feats.get(key, 0)
                col_st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val">{val:.1f if isinstance(val,float) else val}</div>
                    <div class="metric-lbl">{label}</div>
                </div>
                """, unsafe_allow_html=True)

elif run and not smiles_input.strip():
    st.warning("กรุณาใส่ SMILES ก่อน")

# ── Footer note ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1e2d40;
     font-family:'Space Mono',monospace;font-size:0.68rem;color:#334155;
     text-align:center;line-height:2;">
    NeuroLNP v5.1 · Hybrid AI Platform · TIA2026<br>
    ML: Stacking Ensemble (RF + GBR + SVR → Ridge) · B3DB Dataset (n=1,058) · Test R²=0.4783<br>
    Expert Rules: Carbon Tail · PEG Dilemma · Particle Properties · AChR Targeting<br>
    <span style="color:#1e2d40;">─────────────────────────────────────────────</span><br>
    For research purposes only · Not for clinical use
</div>
""", unsafe_allow_html=True)
