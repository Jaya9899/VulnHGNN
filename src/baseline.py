import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from baseline_features import (    # noqa: E402
    build_feature_matrix,
    CLASS_NAMES,
    NUM_CLASSES,
)

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('baseline')

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IR_DIR   = _ROOT / 'data' / 'ir'
DEFAULT_OUT      = _ROOT / 'results' / 'baseline_metrics.json'
RANDOM_STATE     = 42
TEST_SIZE        = 0.20

LABEL_IDS = [0, 2, 3, 17, 18, 25]

def stratified_split(X: pd.DataFrame, y: np.ndarray, test_size: float = TEST_SIZE):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # type: ignore
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(msss.split(X.values, y))
        logger.info("Using iterative stratification (iterstrat)")
    except ImportError:
        logger.warning(
            "iterstrat not installed – falling back to random split. "
            "Install with: pip install iterative-stratification"
        )
        rng = np.random.default_rng(RANDOM_STATE)
        indices = rng.permutation(len(X))
        cut = int(len(X) * (1 - test_size))
        train_idx, test_idx = indices[:cut], indices[cut:]

    X_train = X.iloc[train_idx].values.astype(np.float32)
    X_test  = X.iloc[test_idx].values.astype(np.float32)
    y_train = y[train_idx]
    y_test  = y[test_idx]

    logger.info(
        "Split: %d train / %d test (%.0f%% / %.0f%%)",
        len(train_idx), len(test_idx),
        100 * len(train_idx) / len(X),
        100 * len(test_idx)  / len(X),
    )
    return X_train, X_test, y_train, y_test

def build_model(n_estimators: int = 200) -> OneVsRestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,          # fully grown trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced', # handle class imbalance
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return OneVsRestClassifier(rf, n_jobs=-1)

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    hl = hamming_loss(y_true, y_pred)

    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    micro_p = precision_score(y_true, y_pred, average='micro', zero_division=0)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)

    micro_r = recall_score(y_true, y_pred, average='micro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)

    per_class = {}
    p_arr = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_arr = recall_score   (y_true, y_pred, average=None, zero_division=0)
    f_arr = f1_score       (y_true, y_pred, average=None, zero_division=0)
    sup   = y_true.sum(axis=0).astype(int)

    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            'label_id':  LABEL_IDS[i],
            'index':     i,
            'precision': round(float(p_arr[i]), 4),
            'recall':    round(float(r_arr[i]), 4),
            'f1':        round(float(f_arr[i]), 4),
            'support':   int(sup[i]),
        }

    return {
        'hamming_loss':  round(hl,      4),
        'micro_f1':      round(micro_f1, 4),
        'macro_f1':      round(macro_f1, 4),
        'micro_precision': round(micro_p, 4),
        'macro_precision': round(macro_p, 4),
        'micro_recall':  round(micro_r, 4),
        'macro_recall':  round(macro_r, 4),
        'per_class':     per_class,
    }


def print_report(metrics: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    sep = '─' * 65
    print(f"\n{sep}")
    print(f"  BASELINE CLASSIFIER RESULTS (Random Forest + OneVsRest)")
    print(sep)
    print(f"  Hamming Loss     : {metrics['hamming_loss']:.4f}")
    print(f"  Micro F1         : {metrics['micro_f1']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Micro Precision  : {metrics['micro_precision']:.4f}")
    print(f"  Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"  Micro Recall     : {metrics['micro_recall']:.4f}")
    print(f"  Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<14} {'LabelID':>7} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'Support':>8}")
    print(f"  {'-'*55}")
    for name, m in metrics['per_class'].items():
        print(
            f"  {name:<14} {m['label_id']:>7} "
            f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['support']:>8}"
        )
    print(sep)
    print("\n  Full sklearn classification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=CLASS_NAMES,
            zero_division=0, digits=4,
        )
    )

def run(
    ir_dir: Path,
    out_path: Path,
    n_estimators: int = 200,
    max_samples: Optional[int] = None,
    no_bigrams: bool = False,
) -> dict:
    logger.info("Step 1: Feature Extraction")
    X, y, meta = build_feature_matrix(ir_dir, max_samples=max_samples)

    if no_bigrams:
        uni_cols = [c for c in X.columns if c.startswith('op_')]
        X = X[uni_cols]
        logger.info("Bigram features disabled – using %d unigram features", len(uni_cols))

    logger.info("Feature matrix: %s", X.shape)

    logger.info("\nStep 2: Stratified 80/20 Split")
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    logger.info("\nStep 3: Training RandomForest OneVsRest (%d estimators)",
                n_estimators)
    model = build_model(n_estimators)

    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model.fit(X_train_s, y_train)
    logger.info("Training complete.")

    logger.info("\nStep 4: Evaluation")
    y_pred = model.predict(X_test_s)
    y_pred = np.array(y_pred, dtype=np.float32)

    metrics = evaluate(y_test, y_pred)
    print_report(metrics, y_test, y_pred)

    _log_top_features(model, X, n_top=20)

    logger.info("\nStep 5: Saving Results")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'model': 'RandomForest + OneVsRestClassifier',
        'n_estimators': n_estimators,
        'feature_type': 'bag_of_opcodes_unigram_bigram' if not no_bigrams
                        else 'bag_of_opcodes_unigram',
        'n_features': int(X.shape[1]),
        'n_train': int(X_train.shape[0]),
        'n_test':  int(X_test.shape[0]),
        'label_mapping': {
            name: {'label_id': lid, 'index': idx}
            for idx, (name, lid) in enumerate(zip(CLASS_NAMES, LABEL_IDS))
        },
        'metrics': metrics,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved → %s", out_path)
    return output

def _log_top_features(model: OneVsRestClassifier, X: pd.DataFrame, n_top: int = 20) -> None:
    try:
        importances = np.zeros(X.shape[1])
        for estimator in model.estimators_:
            importances += estimator.feature_importances_
        importances /= len(model.estimators_)

        top_idx = np.argsort(importances)[::-1][:n_top]
        logger.info("\nTop %d features by mean importance:", n_top)
        for rank, i in enumerate(top_idx, 1):
            logger.info("  %2d. %-30s  %.5f", rank, X.columns[i], importances[i])
    except Exception as exc:
        logger.debug("Could not extract feature importances: %s", exc)

def _parse_args():
    p = argparse.ArgumentParser(
        description='Bag-of-Opcodes Random Forest baseline for LLVM IR vulnerability detection.'
    )
    p.add_argument('--ir_dir',       type=Path, default=DEFAULT_IR_DIR)
    p.add_argument('--out',          type=Path, default=DEFAULT_OUT)
    p.add_argument('--n_estimators', type=int,  default=200,
                   help='Number of trees in each RF binary classifier (default 200)')
    p.add_argument('--max_samples',  type=int,  default=None,
                   help='Cap number of .ll files (for quick testing)')
    p.add_argument('--no_bigrams',   action='store_true',
                   help='Disable bigram features (use unigrams only)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run(
        ir_dir=args.ir_dir,
        out_path=args.out,
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        no_bigrams=args.no_bigrams,
    )
