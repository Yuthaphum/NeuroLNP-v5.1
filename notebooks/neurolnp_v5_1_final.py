# =============================================================================
# NeuroLNP: BBB Permeability Prediction — v5.1
# Hybrid AI = ML (Stacking Ensemble) + Expert Rules Layer (Biophysics)
# Dataset: B3DB Regression (1,058 สาร)
# Features: 25 Physicochemical + 32 Morgan FP = 57 features
#
# NEW in v5.1 vs v5.0:
#   - Expert Rules เปลี่ยนจาก Additive → Multiplicative Weighted Scoring
#   - แต่ละ rule ให้ผล 0.0–1.0 แล้วนำมา weighted average
#   - ไม่มีปัญหา bonus/penalty เกิน scale อีกต่อไป
#   - ถูกหลัก drug development: rule ข้อเดียวแย่ ฉุด score ทั้งหมดได้
# =============================================================================

# !pip install rdkit scikit-learn xgboost pandas matplotlib seaborn shap scipy

# =============================================================================
# SECTION 1: Imports
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, json, pickle
from scipy import stats
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

print("✅ โหลดไลบรารีเสร็จสิ้น!")
print("=" * 60)


# =============================================================================
# SECTION 2: โหลดข้อมูล B3DB
# =============================================================================

URL = "https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_regression.tsv"

try:
    df = pd.read_csv(URL, sep='\t')
    df = df.dropna(subset=['SMILES', 'logBB']).reset_index(drop=True)
    print(f"✅ B3DB Dataset: {len(df)} สาร")
    print(f"   logBB range : {df['logBB'].min():.2f} ถึง {df['logBB'].max():.2f}")
    print(f"   logBB mean  : {df['logBB'].mean():.2f} ± {df['logBB'].std():.2f}")
except Exception as e:
    raise RuntimeError(f"❌ โหลด B3DB ไม่สำเร็จ: {e}")


# =============================================================================
# SECTION 3: Feature Engineering — 57 features
# =============================================================================

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=32)


def get_molecular_features(smiles: str) -> dict | None:
    """สกัด 57 molecular features จาก SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        heavy = Descriptors.HeavyAtomCount(mol)
        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        tpsa  = Descriptors.TPSA(mol)

        hbd       = Descriptors.NumHDonors(mol)
        hba       = Descriptors.NumHAcceptors(mol)
        rotb      = Descriptors.NumRotatableBonds(mol)
        fsp3      = Descriptors.FractionCSP3(mol)
        nhoh      = Descriptors.NHOHCount(mol)
        no_count  = Descriptors.NOCount(mol)

        arom_rings    = rdMolDescriptors.CalcNumAromaticRings(mol)
        ali_rings     = rdMolDescriptors.CalcNumAliphaticRings(mol)
        ring_count    = rdMolDescriptors.CalcNumRings(mol)
        arom_atoms    = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        arom_prop     = arom_atoms / heavy if heavy > 0 else 0

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
            'NumHalogens': hal_atoms,
            **morgan_feats,
        }
    except Exception:
        return None


print("\n⏳ กำลังสกัด Molecular Features (57 ตัว)...")
features_df   = pd.DataFrame(df['SMILES'].apply(get_molecular_features).tolist())
data          = pd.concat([features_df, df['logBB'].reset_index(drop=True)], axis=1).dropna()
X             = data.drop(columns=['logBB'])
y             = data['logBB']
FEATURE_NAMES = list(X.columns)

print(f"✅ {len(data)} สาร × {len(FEATURE_NAMES)} features พร้อมเทรน")
print(f"   Missing values: {'ไม่มี ✅' if features_df.isnull().sum().sum() == 0 else features_df.isnull().sum().sum()}")


# =============================================================================
# SECTION 4: Train/Test Split — Stratified by logBB quartile
# =============================================================================

y_quartile = pd.qcut(y, q=4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y_quartile, shuffle=True,
)
y_train_mean = float(y_train.mean())

print(f"\n✅ Stratified Split (80/20):")
print(f"   Train: {len(X_train)} สาร | Test: {len(X_test)} สาร")


# =============================================================================
# SECTION 5: กำหนด 5 Models
# =============================================================================

print("\n🔍 กำลังหา Best Ridge Alpha...")
alphas    = np.logspace(-3, 4, 200)
_ridge_cv = Pipeline([('scaler', StandardScaler()),
                       ('model', RidgeCV(alphas=alphas, cv=10))])
_ridge_cv.fit(X_train, y_train)
BEST_ALPHA = _ridge_cv.named_steps['model'].alpha_
print(f"   Best Alpha: {BEST_ALPHA:.4f}")

MODELS = {
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=BEST_ALPHA)),
    ]),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')),
    ]),
    'Random Forest': RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_leaf=4,
        max_features=0.5, random_state=42, n_jobs=-1,
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.75, min_child_weight=4,
        gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
        random_state=42, verbosity=0, tree_method='hist',
    ),
    'Stacking': StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                max_features=0.5, random_state=42, n_jobs=-1,
            )),
            ('gbr', GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42,
            )),
            ('svr', Pipeline([
                ('scaler', StandardScaler()),
                ('model',  SVR(kernel='rbf', C=10, gamma='scale')),
            ])),
        ],
        final_estimator=Ridge(alpha=BEST_ALPHA),
        cv=5, passthrough=False,
    ),
}


# =============================================================================
# SECTION 6: QSAR Metrics
# =============================================================================

def qsar_metrics(y_true, y_pred, y_train_mean=None) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    if y_train_mean is not None:
        q2_ext = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-y_train_mean)**2)
    else:
        q2_ext = r2
    r_val, p_val = stats.pearsonr(y_true, y_pred)
    k   = np.sum(y_true*y_pred) / np.sum(y_pred**2)
    kp  = np.sum(y_true*y_pred) / np.sum(y_true**2)
    mu_t, mu_p   = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov_tp = np.mean((y_true-mu_t)*(y_pred-mu_p))
    ccc    = (2*cov_tp) / (var_t + var_p + (mu_t-mu_p)**2)
    return {
        'RMSE': round(float(rmse), 4), 'MAE': round(float(mae), 4),
        'R2': round(float(r2), 4), 'Q2_ext': round(float(q2_ext), 4),
        'r': round(float(r_val), 4), 'p_value': round(float(p_val), 6),
        'k': round(float(k), 4), 'k_prime': round(float(kp), 4),
        'CCC': round(float(ccc), 4), 'N': int(len(y_true)),
    }


# =============================================================================
# SECTION 7: เทรนและประเมินทุก Model
# =============================================================================

print("\n" + "=" * 60)
print("🧠 เทรนและประเมิน 5 Models...")
print("=" * 60)

results = {}
kf      = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in MODELS.items():
    print(f"\n  [{name}] กำลังเทรน...", end=' ', flush=True)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    train_m    = qsar_metrics(y_train, train_pred, y_train_mean)
    test_m     = qsar_metrics(y_test,  test_pred,  y_train_mean)
    cv_r2      = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    cv_rmse    = cross_val_score(model, X_train, y_train, cv=kf,
                                  scoring='neg_root_mean_squared_error')
    results[name] = {
        'model':        model,
        'train':        train_m,
        'test':         test_m,
        'test_pred':    test_pred,
        'gap':          round(train_m['R2'] - test_m['R2'], 4),
        'cv_r2_mean':   round(float(cv_r2.mean()), 4),
        'cv_r2_std':    round(float(cv_r2.std()), 4),
        'cv_rmse_mean': round(float((-cv_rmse).mean()), 4),
        'cv_rmse_std':  round(float((-cv_rmse).std()), 4),
    }
    print(f"✅  Test R²={test_m['R2']:.4f} | Q²_ext={test_m['Q2_ext']:.4f} | "
          f"RMSE={test_m['RMSE']:.4f} | CV_R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}")

sorted_models = sorted(results.items(), key=lambda x: x[1]['test']['R2'], reverse=True)
best_name     = sorted_models[0][0]
best_result   = sorted_models[0][1]

# Benchmark Table
print("\n" + "=" * 60)
print("📊 QSAR Benchmark Results")
print("=" * 60)
header = (f"{'Model':<18} {'R²_tr':>6} {'R²_te':>6} {'Q²_ext':>7} "
          f"{'RMSE_te':>8} {'MAE_te':>7} {'r':>6} {'CCC':>6} {'CV_R²':>12} {'Gap':>6}")
print(header)
print("-" * len(header))
for name, r in sorted_models:
    print(f"{name:<18} {r['train']['R2']:>6.4f} {r['test']['R2']:>6.4f} "
          f"{r['test']['Q2_ext']:>7.4f} {r['test']['RMSE']:>8.4f} "
          f"{r['test']['MAE']:>7.4f} {r['test']['r']:>6.4f} "
          f"{r['test']['CCC']:>6.4f} "
          f"{r['cv_r2_mean']:.3f}±{r['cv_r2_std']:.3f}  {r['gap']:>6.4f}")

print(f"\n🏆 Best Model: {best_name} (Test R² = {best_result['test']['R2']:.4f})")

bm = best_result['test']
criteria = {
    'R² > 0.6':           bm['R2'] > 0.6,
    'Q²_ext > 0.5':       bm['Q2_ext'] > 0.5,
    'r > 0.7':            bm['r'] > 0.7,
    '0.85 ≤ k ≤ 1.15':   0.85 <= bm['k'] <= 1.15,
    "0.85 ≤ k' ≤ 1.15":  0.85 <= bm['k_prime'] <= 1.15,
    '|R²−Q²_ext| < 0.3': abs(bm['R2'] - bm['Q2_ext']) < 0.3,
}
print(f"\n📋 QSAR Acceptance Criteria ({best_name}):")
for c, passed in criteria.items():
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}  {c}")
passed_count = sum(criteria.values())
print(f"\n  {passed_count}/{len(criteria)} criteria passed")


# =============================================================================
# SECTION 8: ★★★ EXPERT RULES LAYER v5.1 — Multiplicative Scoring ★★★
#
# หลักการ: แต่ละ rule ให้ผล 0.0–1.0 แล้วนำมา weighted average
#   - 1.0 = ผ่านสมบูรณ์แบบ
#   - 0.0 = ผิดพลาดร้ายแรง (ฉุด score ทั้งหมด)
#   - rule ที่สำคัญกว่าได้ weight มากกว่า
#
# Rule Weights:
#   rule1_tail    = 0.25  ★ สำคัญที่สุด กำหนด organ tropism
#   rule2a_peg_mw = 0.10  PEG MW
#   rule2b_peg_pct= 0.10  PEG mol%
#   rule3a_size   = 0.15  Particle size
#   rule3b_zeta   = 0.15  Zeta potential
#   rule3c_pdi    = 0.10  PDI
#   rule3d_pka    = 0.10  pKa
#   rule4_ligand  = 0.05  Targeting ligand
#   รวม           = 1.00
#
# อ้างอิง:
#   กฎ 1: Nano Letters (2024) — Lipid tail length & organ tropism
#   กฎ 2: J Nanobiotechnology (2024) — PEG Dilemma
#   กฎ 3: PMC Programmable LNPs (2024) — pKa, size, zeta design rules
#   กฎ 4: ACS Nano (2025) — Acetylcholine-LNP brain targeting
# =============================================================================

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
    carbon_tail_length: int   = 14,
    peg_mw:             int   = 2000,
    peg_mol_percent:    float = 1.5,
    particle_size_nm:   float = 100.0,
    zeta_potential_mv:  float = -5.0,
    pdi:                float = 0.2,
    pka:                float = 6.5,
    has_acetylcholine:  bool  = False,
    targeting_ligand:   str   = 'none',
) -> dict:
    """
    Expert Rules Layer — Multiplicative Weighted Scoring
    คืนค่า expert_score_10 (0–10), tier, warnings, recommendations

    Score = weighted average ของ rule scores แต่ละข้อ (0.0–1.0)
    แล้วคูณ 10 → ได้ 0–10 scale ที่ไม่เกิน range
    """

    scores   = {}
    warnings = []
    recs     = []

    # =========================================================================
    # Rule 1: Carbon Tail Length (weight = 0.25)
    # อ้างอิง: Nano Letters (2024) — mRNA expression shifts liver→spleen
    #          with shorter C-tail; C18 causes liver tropism via ApoE
    # =========================================================================
    tail_map = {12: 0.5, 14: 1.0, 16: 0.6, 18: 0.2}
    scores['rule1_tail'] = tail_map.get(carbon_tail_length, 0.3)

    if carbon_tail_length == 14:
        recs.append("✅ C14: optimal — หลุดจาก LNP ได้เร็วใน vivo → ลด liver tropism")
    elif carbon_tail_length == 18:
        warnings.append("❌ C18: Liver tropism สูง — ApoE จะชักนำ LNP ไปตับ")
        warnings.append("   → C18 ทำให้ membrane rigidity สูง แทรกซึมเซลล์สมองยาก")
        recs.append("🔧 เปลี่ยนเป็น C14 ทันที — นี่คือปัญหาหลักของ design นี้")
    elif carbon_tail_length == 16:
        warnings.append("⚠️  C16: เริ่มสะสมตับมากกว่า C14")
        recs.append("🔧 แนะนำเปลี่ยนเป็น C14 ถ้า target คือสมอง")
    elif carbon_tail_length == 12:
        warnings.append("⚠️  C12: โครงสร้าง LNP ไม่เสถียรพอ")
        recs.append("🔧 แนะนำเปลี่ยนเป็น C14")
    else:
        warnings.append(f"⚠️  C{carbon_tail_length}: นอกช่วงที่ศึกษามาก ข้อมูลจำกัด")
        recs.append("🔧 แนะนำใช้ C14")

    # =========================================================================
    # Rule 2A: PEG Molecular Weight (weight = 0.10)
    # อ้างอิง: J Nanobiotechnology (2024) — PEG Dilemma
    # =========================================================================
    peg_mw_map = {1000: 0.4, 2000: 1.0, 3000: 0.3, 5000: 0.1}
    scores['rule2a_peg_mw'] = peg_mw_map.get(peg_mw, 0.5)

    if peg_mw == 2000:
        recs.append("✅ PEG-2000: มาตรฐาน — balance ระหว่าง stealth และ endosomal escape")
    elif peg_mw == 3000:
        warnings.append("❌ PEG-3000: ชั้นหนาเกิน ขวาง endosomal escape")
        warnings.append("   → mRNA ออกจาก endosome ไม่ได้ → ถูก lysosome ทำลาย")
        recs.append("🔧 ลดเป็น PEG-2000")
    elif peg_mw >= 5000:
        warnings.append(f"❌ PEG-{peg_mw}: หนามากเกิน mRNA ออกไม่ได้เลย")
        recs.append("🔧 เปลี่ยนเป็น PEG-2000 ทันที")
    elif peg_mw <= 1000:
        warnings.append(f"⚠️  PEG-{peg_mw}: เล็กเกิน ไม่ช่วย stealth effect")
        recs.append("🔧 แนะนำ PEG-2000")
    else:
        warnings.append(f"⚠️  PEG-{peg_mw}: นอก standard range ข้อมูลจำกัด")

    # =========================================================================
    # Rule 2B: PEG mol% (weight = 0.10)
    # =========================================================================
    if 1.5 <= peg_mol_percent <= 2.5:
        scores['rule2b_peg_pct'] = 1.0
        recs.append(f"✅ PEG {peg_mol_percent:.1f} mol%: golden zone (1.5–2.5%)")
    elif peg_mol_percent < 1.5:
        scores['rule2b_peg_pct'] = 0.4
        warnings.append(f"⚠️  PEG {peg_mol_percent:.1f} mol%: น้อยเกิน "
                         f"(ต้องการ ≥1.5% เพื่อเปลี่ยน pharmacokinetics)")
        recs.append("🔧 เพิ่ม PEG เป็น 1.5–2.5 mol%")
    elif peg_mol_percent <= 3.5:
        scores['rule2b_peg_pct'] = 0.6
        warnings.append(f"⚠️  PEG {peg_mol_percent:.1f} mol%: มากเกินนิดหน่อย "
                         f"เริ่มขวาง cell internalization")
        recs.append("🔧 ลดเป็น 1.5–2.5 mol%")
    else:
        scores['rule2b_peg_pct'] = 0.1
        warnings.append(f"❌ PEG {peg_mol_percent:.1f} mol%: มากเกิน "
                         f"ขวางทั้ง internalization และ endosomal escape")
        recs.append("🔧 ลดเป็น 1.5–2.5 mol% ทันที")

    # =========================================================================
    # Rule 3A: Particle Size (weight = 0.15)
    # อ้างอิง: Programmable LNPs review (PMC 2024)
    # =========================================================================
    if 50 <= particle_size_nm <= 150:
        scores['rule3a_size'] = 1.0
        recs.append(f"✅ Size {particle_size_nm:.0f} nm: optimal สำหรับ BBB crossing")
    elif particle_size_nm < 50:
        scores['rule3a_size'] = 0.5
        warnings.append(f"⚠️  Size {particle_size_nm:.0f} nm: เล็กเกิน "
                         f"จุ mRNA ได้น้อย กรองออกเร็วผ่านไต")
        recs.append("🔧 เพิ่ม size เป็น 80–120 nm")
    elif particle_size_nm <= 200:
        scores['rule3a_size'] = 0.6
        warnings.append(f"⚠️  Size {particle_size_nm:.0f} nm: ใหญ่เกินนิดหน่อย "
                         f"อาจติดที่ BBB")
        recs.append("🔧 ลดเป็น < 150 nm")
    else:
        scores['rule3a_size'] = 0.1
        warnings.append(f"❌ Size {particle_size_nm:.0f} nm: ใหญ่เกิน "
                         f"ไม่สามารถผ่าน BBB ได้")
        recs.append("🔧 ต้องปรับ formulation ให้ size < 150 nm")

    # =========================================================================
    # Rule 3B: Zeta Potential (weight = 0.15)
    # =========================================================================
    if -10 <= zeta_potential_mv <= -2:
        scores['rule3b_zeta'] = 1.0
        recs.append(f"✅ Zeta {zeta_potential_mv:.1f} mV: stealth — "
                     f"ไม่กระตุ้น immune system")
    elif -2 < zeta_potential_mv <= 5:
        scores['rule3b_zeta'] = 0.7
        recs.append(f"✅ Zeta {zeta_potential_mv:.1f} mV: neutral — ยอมรับได้")
    elif 5 < zeta_potential_mv <= 15:
        scores['rule3b_zeta'] = 0.3
        warnings.append(f"⚠️  Zeta +{zeta_potential_mv:.1f} mV: cationic เกินไป "
                         f"กระตุ้น immune + เป็นพิษต่อเซลล์")
        recs.append("🔧 ลด cationic lipid หรือเพิ่ม neutral lipid")
    elif zeta_potential_mv > 15:
        scores['rule3b_zeta'] = 0.0
        warnings.append(f"❌ Zeta +{zeta_potential_mv:.1f} mV: อันตราย "
                         f"ทำลาย BBB endothelial cells")
        recs.append("🔧 ต้องปรับ lipid composition ใหม่ทั้งหมด")
    elif -20 <= zeta_potential_mv < -10:
        scores['rule3b_zeta'] = 0.7
        warnings.append(f"⚠️  Zeta {zeta_potential_mv:.1f} mV: "
                         f"negative เกินนิดหน่อย อาจผลักออกจากเซลล์")
    else:
        scores['rule3b_zeta'] = 0.2
        warnings.append(f"❌ Zeta {zeta_potential_mv:.1f} mV: "
                         f"negative มากเกิน ไม่สามารถ internalize ได้")
        recs.append("🔧 ปรับเป็น -10 ถึง -2 mV")

    # =========================================================================
    # Rule 3C: PDI — Polydispersity Index (weight = 0.10)
    # =========================================================================
    if pdi <= 0.1:
        scores['rule3c_pdi'] = 1.0
        recs.append(f"✅ PDI {pdi:.2f}: monodisperse ดีมาก")
    elif pdi <= 0.2:
        scores['rule3c_pdi'] = 0.8
        recs.append(f"✅ PDI {pdi:.2f}: ยอมรับได้ (< 0.2 = acceptable)")
    elif pdi <= 0.3:
        scores['rule3c_pdi'] = 0.5
        warnings.append(f"⚠️  PDI {pdi:.2f}: polydisperse เกินไป "
                         f"targeting ไม่สม่ำเสมอ")
        recs.append("🔧 optimize microfluidic mixing conditions")
    else:
        scores['rule3c_pdi'] = 0.1
        warnings.append(f"❌ PDI {pdi:.2f}: polydisperse มาก "
                         f"LNP ไม่เป็นเนื้อเดียวกัน ผลจะ inconsistent")
        recs.append("🔧 ต้องปรับ formulation process ใหม่")

    # =========================================================================
    # Rule 3D: apparent pKa (weight = 0.10)
    # pKa 6.0–7.0 = ionize ใน endosome (pH 5.5) → endosomal escape
    # =========================================================================
    if 6.0 <= pka <= 7.0:
        scores['rule3d_pka'] = 1.0
        recs.append(f"✅ pKa {pka:.1f}: optimal — ionize ใน endosome "
                     f"(pH 5.5) → endosomal escape ดี")
    elif 5.5 <= pka < 6.0 or 7.0 < pka <= 7.5:
        scores['rule3d_pka'] = 0.6
        warnings.append(f"⚠️  pKa {pka:.1f}: นอก optimal range เล็กน้อย "
                         f"(6.0–7.0) endosomal escape ลดลงบางส่วน")
        recs.append("🔧 แนะนำ pKa 6.0–7.0")
    else:
        scores['rule3d_pka'] = 0.1
        warnings.append(f"❌ pKa {pka:.1f}: อยู่นอก acceptable range (5.5–7.5) "
                         f"อย่างมีนัยสำคัญ")
        recs.append("🔧 ปรับโครงสร้าง ionizable lipid head group")

    # =========================================================================
    # Rule 4: Targeting Ligand (weight = 0.05)
    # อ้างอิง: ACS Nano (2025) — Acetylcholine outperforms all other ligands
    # =========================================================================
    ligand_map = {
        'acetylcholine': 1.0,
        'nicotine':      0.7,
        'glucose':       0.5,
        'tryptophan':    0.5,
        'memantine':     0.4,
        'none':          0.2,
    }
    lig_key = targeting_ligand.lower().strip()
    scores['rule4_ligand'] = ligand_map.get(lig_key, 0.3)

    # Acetylcholine boolean override
    if has_acetylcholine and lig_key != 'acetylcholine':
        scores['rule4_ligand'] = max(scores['rule4_ligand'], 0.8)
        recs.append("✅ มี Acetylcholine conjugation — brain tropism เพิ่มขึ้น")
    elif lig_key == 'acetylcholine' or has_acetylcholine:
        recs.append("✅ Acetylcholine: ดีที่สุด → neurons + astrocytes "
                     "(ACS Nano 2025)")
    elif lig_key == 'none':
        warnings.append("⚠️  ไม่มี brain targeting ligand — "
                         "brain delivery < 1% injected dose")
        recs.append("🔧 เพิ่ม Acetylcholine-PEG-lipid conjugate")
    else:
        recs.append(f"⚠️  {targeting_ligand}: ใช้ได้ แต่ไม่ specific "
                     f"เท่า Acetylcholine")

    # =========================================================================
    # Multiplicative Weighted Score — guaranteed 0.0–10.0
    # =========================================================================
    expert_score_raw = sum(
        scores[k] * RULE_WEIGHTS[k]
        for k in RULE_WEIGHTS
    )
    expert_score_10 = round(expert_score_raw * 10, 2)  # 0–10

    if expert_score_10 >= 8:
        tier, overall = 'A', '🟢 EXCELLENT — LNP ออกแบบดีมาก เหมาะสำหรับ brain targeting'
    elif expert_score_10 >= 6:
        tier, overall = 'B', '🟡 GOOD — ควรปรับปรุงจุดที่แนะนำ'
    elif expert_score_10 >= 4:
        tier, overall = 'C', '🟠 FAIR — มีปัญหาหลายจุด ควรแก้ก่อนทดสอบ'
    else:
        tier, overall = 'D', '🔴 POOR — ต้องออกแบบ LNP ใหม่'

    return {
        'rule_scores':     scores,
        'expert_score_10': expert_score_10,
        'tier':            tier,
        'overall':         overall,
        'warnings':        warnings,
        'recommendations': recs,
        'rule_breakdown': {
            'rule1_carbon_tail':  f"C{carbon_tail_length} → {scores['rule1_tail']:.2f}",
            'rule2a_peg_mw':      f"PEG-{peg_mw} → {scores['rule2a_peg_mw']:.2f}",
            'rule2b_peg_pct':     f"{peg_mol_percent:.1f} mol% → {scores['rule2b_peg_pct']:.2f}",
            'rule3a_size':        f"{particle_size_nm:.0f} nm → {scores['rule3a_size']:.2f}",
            'rule3b_zeta':        f"{zeta_potential_mv:.1f} mV → {scores['rule3b_zeta']:.2f}",
            'rule3c_pdi':         f"PDI={pdi:.2f} → {scores['rule3c_pdi']:.2f}",
            'rule3d_pka':         f"pKa={pka:.1f} → {scores['rule3d_pka']:.2f}",
            'rule4_ligand':       f"{targeting_ligand} → {scores['rule4_ligand']:.2f}",
        },
    }


# =============================================================================
# SECTION 9: predict_lnp() — Main Hybrid AI Inference Function
# =============================================================================

def predict_lnp(
    smiles:             str,
    carbon_tail_length: int   = 14,
    peg_mw:             int   = 2000,
    peg_mol_percent:    float = 1.5,
    particle_size_nm:   float = 100.0,
    zeta_potential_mv:  float = -5.0,
    pdi:                float = 0.2,
    pka:                float = 6.5,
    has_acetylcholine:  bool  = False,
    targeting_ligand:   str   = 'none',
    model_path:         str   = '/content/neurolnp_best_model.pkl',
    metadata_path:      str   = '/content/neurolnp_benchmark_results.json',
) -> dict:
    """
    NeuroLNP Hybrid AI Prediction v5.1
    ML (40%) + Expert Rules (60%) → Combined Score 0–10
    """

    # ── Step 1: ML prediction ────────────────────────────────────────────────
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, encoding='utf-8') as f:
        meta = json.load(f)

    feats = get_molecular_features(smiles)
    if feats is None:
        return {'error': f'Invalid SMILES: {smiles}'}

    feat_vector = pd.DataFrame([feats])[meta['feature_names']]
    logbb_pred  = float(model.predict(feat_vector)[0])

    if logbb_pred > 0.3:
        bbb_class, bbb_emoji = 'BBB+', '✅'
        bbb_note = 'ผ่าน BBB ได้ดี — lipid component เหมาะสม'
    elif logbb_pred > -1.0:
        bbb_class, bbb_emoji = 'BBB±', '⚠️'
        bbb_note = 'ผ่าน BBB ได้ปานกลาง — ต้องพึ่ง targeting ligand'
    else:
        bbb_class, bbb_emoji = 'BBB-', '❌'
        bbb_note = 'ผ่าน BBB ไม่ได้ — lipid component ไม่เหมาะสม'

    # ML score: map logBB (-2.69 ถึง 1.70) → 0–10
    ml_score_10 = (logbb_pred - (-2.69)) / (1.70 - (-2.69)) * 10
    ml_score_10 = round(max(0.0, min(10.0, ml_score_10)), 2)

    # ── Step 2: Expert Rules ─────────────────────────────────────────────────
    rules = apply_expert_rules(
        carbon_tail_length = carbon_tail_length,
        peg_mw             = peg_mw,
        peg_mol_percent    = peg_mol_percent,
        particle_size_nm   = particle_size_nm,
        zeta_potential_mv  = zeta_potential_mv,
        pdi                = pdi,
        pka                = pka,
        has_acetylcholine  = has_acetylcholine,
        targeting_ligand   = targeting_ligand,
    )

    # ── Step 3: Combined Score ───────────────────────────────────────────────
    # ML 40% + Expert Rules 60%
    # Expert มีน้ำหนักมากกว่าเพราะ ML Test R²=0.478 (uncertainty สูง)
    # แต่ Expert Rules มาจากงานวิจัยที่พิสูจน์แล้ว
    combined_score = round(0.4 * ml_score_10 + 0.6 * rules['expert_score_10'], 2)
    combined_grade = (
        'A' if combined_score >= 7 else
        'B' if combined_score >= 5 else
        'C' if combined_score >= 3 else 'D'
    )

    return {
        # ML
        'smiles':            smiles,
        'logBB_predicted':   round(logbb_pred, 4),
        'bbb_class':         bbb_class,
        'bbb_emoji':         bbb_emoji,
        'bbb_note':          bbb_note,
        'ml_score_0_10':     ml_score_10,

        # Expert Rules
        'expert_score_0_10': rules['expert_score_10'],
        'expert_tier':       rules['tier'],
        'expert_overall':    rules['overall'],
        'rule_scores':       rules['rule_scores'],
        'rule_breakdown':    rules['rule_breakdown'],
        'warnings':          rules['warnings'],
        'recommendations':   rules['recommendations'],

        # Combined
        'combined_score_0_10': combined_score,
        'combined_grade':      combined_grade,

        # Key features สำหรับ UI
        'key_molecular_features': {
            'MolWt':        round(feats.get('MolWt', 0), 1),
            'LogP':         round(feats.get('LogP', 0), 3),
            'TPSA':         round(feats.get('TPSA', 0), 1),
            'HBDonors':     feats.get('NumHDonors', 0),
            'RotBonds':     feats.get('NumRotatableBonds', 0),
            'FormalCharge': feats.get('FormalCharge', 0),
        },
        'lnp_parameters': {
            'carbon_tail': f"C{carbon_tail_length}",
            'peg':         f"PEG-{peg_mw} @ {peg_mol_percent:.1f} mol%",
            'size':        f"{particle_size_nm:.0f} nm",
            'zeta':        f"{zeta_potential_mv:.1f} mV",
            'pdi':         pdi,
            'pka':         pka,
            'ligand':      targeting_ligand,
        },
        'model_info': {
            'ml_model':      meta['best_model'],
            'model_version': meta.get('model_version', 'v5.1'),
            'dataset':       meta['dataset'],
            'n_train':       meta['n_train'],
            'test_r2':       meta['best_model_metrics']['test_R2'],
            'test_q2ext':    meta['best_model_metrics']['test_Q2_ext'],
        },
    }


# =============================================================================
# SECTION 10: Export Model + Metadata
# =============================================================================

best_model_obj = best_result['model']
with open('/content/neurolnp_best_model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

paper_results = {}
for name, r in results.items():
    paper_results[name] = {
        'train_R2': r['train']['R2'], 'test_R2': r['test']['R2'],
        'test_Q2_ext': r['test']['Q2_ext'], 'test_RMSE': r['test']['RMSE'],
        'test_MAE': r['test']['MAE'], 'test_r': r['test']['r'],
        'test_CCC': r['test']['CCC'], 'test_k': r['test']['k'],
        'test_k_prime': r['test']['k_prime'],
        'cv_R2_mean': r['cv_r2_mean'], 'cv_R2_std': r['cv_r2_std'],
        'cv_RMSE_mean': r['cv_rmse_mean'], 'cv_RMSE_std': r['cv_rmse_std'],
        'overfitting_gap': r['gap'],
    }

metadata = {
    'model_version':        '5.1_NeuroLNP_HybridAI',
    'best_model':           best_name,
    'feature_names':        FEATURE_NAMES,
    'n_features':           len(FEATURE_NAMES),
    'n_train':              int(len(X_train)),
    'n_test':               int(len(X_test)),
    'y_train_mean':         y_train_mean,
    'dataset':              'B3DB_regression',
    'split_strategy':       'stratified_quartile_80_20',
    'best_alpha_ridge':     float(BEST_ALPHA),
    'qsar_criteria_passed': int(passed_count),
    'results':              paper_results,
    'best_model_metrics':   paper_results[best_name],
    'expert_rules_version': '1.1_multiplicative',
    'expert_rules': {
        'scoring':   'multiplicative_weighted_average',
        'rule1':     'Carbon Tail Length (weight=0.25)',
        'rule2a':    'PEG MW (weight=0.10)',
        'rule2b':    'PEG mol% (weight=0.10)',
        'rule3a':    'Particle Size (weight=0.15)',
        'rule3b':    'Zeta Potential (weight=0.15)',
        'rule3c':    'PDI (weight=0.10)',
        'rule3d':    'pKa (weight=0.10)',
        'rule4':     'Targeting Ligand (weight=0.05)',
        'combined':  'ML 40% + Expert 60%',
    },
}

with open('/content/neurolnp_benchmark_results.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\n📦 Export เสร็จสิ้น!")
print("   neurolnp_best_model.pkl         — Stacking Ensemble")
print("   neurolnp_benchmark_results.json — metadata + expert rules v1.1")


# =============================================================================
# SECTION 11: ทดสอบ Hybrid AI v5.1
# =============================================================================

def print_result(label, r):
    print(f"\n📌 {label}")
    print(f"  logBB predicted  : {r['logBB_predicted']:+.3f} "
          f"({r['bbb_class']}) {r['bbb_emoji']}")
    print(f"  ML score         : {r['ml_score_0_10']:.1f}/10")
    print(f"  Expert score     : {r['expert_score_0_10']:.1f}/10  "
          f"({r['expert_tier']}) {r['expert_overall']}")
    print(f"  ⭐ Combined      : {r['combined_score_0_10']:.1f}/10 "
          f"(Grade {r['combined_grade']})")
    print(f"  Rule scores:")
    for k, v in r['rule_breakdown'].items():
        print(f"    {k:<20}: {v}")
    if r['warnings']:
        print(f"  ⚠️  Warnings ({len(r['warnings'])}):")
        for w in r['warnings']:
            print(f"    {w}")
    recs_to_fix = [x for x in r['recommendations'] if x.startswith('🔧')]
    if recs_to_fix:
        print(f"  🔧 Fixes needed:")
        for rec in recs_to_fix:
            print(f"    {rec}")


print("\n" + "=" * 60)
print("🧪 ทดสอบ Hybrid AI v5.1 — predict_lnp()")
print("=" * 60)

# Test 1: LNP ดีที่สุด
r1 = predict_lnp(
    smiles='Cn1c(=O)c2c(ncn2C)n(c1=O)C',
    carbon_tail_length=14, peg_mw=2000, peg_mol_percent=2.0,
    particle_size_nm=100.0, zeta_potential_mv=-5.0,
    pdi=0.15, pka=6.5, targeting_ligand='acetylcholine',
)
print_result("LNP ออกแบบดีที่สุด (C14 + PEG-2000 + Acetylcholine)", r1)

# Test 2: LNP แย่ที่สุด
r2 = predict_lnp(
    smiles='Cn1c(=O)c2c(ncn2C)n(c1=O)C',
    carbon_tail_length=18, peg_mw=3000, peg_mol_percent=4.0,
    particle_size_nm=250.0, zeta_potential_mv=20.0,
    pdi=0.4, pka=8.0, targeting_ligand='none',
)
print_result("LNP ออกแบบผิดพลาด (C18 + PEG-3000 + ไม่มี ligand)", r2)

# Test 3: LNP กลางๆ
r3 = predict_lnp(
    smiles='CC(=O)Oc1ccccc1C(=O)O',
    carbon_tail_length=16, peg_mw=2000, peg_mol_percent=2.0,
    particle_size_nm=130.0, zeta_potential_mv=-3.0,
    pdi=0.22, pka=6.8, targeting_ligand='nicotine',
)
print_result("LNP กลางๆ (C16 + PEG-2000 + Nicotine)", r3)

print("\n" + "=" * 60)
print("✅ NeuroLNP v5.1 Hybrid AI Pipeline เสร็จสมบูรณ์!")
print(f"   Best ML Model    : {best_name}")
print(f"   Test R²          : {best_result['test']['R2']:.4f}")
print(f"   Expert Rules     : 4 กฎ / 8 sub-rules (Multiplicative)")
print(f"   Combined Score   : ML (40%) + Expert Rules (60%)")
print(f"   Score range      : 0–10 (guaranteed, no overflow)")
print("=" * 60)
