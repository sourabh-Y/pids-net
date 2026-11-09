"""Evaluation: accuracy, weighted F1, MCC, one-vs-rest AUC, confusion matrices,
LLL/LLLG-merged fault-type accuracy and accuracy by fault-resistance bin."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

RF_BINS = [(0.5, 5), (5, 20), (20, 40), (40, 70), (70, 100)]


def merge_lll(y):
    """ABCG (11) -> ABC (10): a symmetric 3-phase fault draws no ground current in a balanced
    system, so the two classes are indistinguishable from phase currents."""
    return np.where(np.asarray(y) == 11, 10, y)


def scores(y, prob):
    yh = prob.argmax(1)
    out = dict(acc=accuracy_score(y, yh), f1=f1_score(y, yh, average='weighted'),
               mcc=matthews_corrcoef(y, yh))
    try:
        out['auc'] = roc_auc_score(y, prob, multi_class='ovr', labels=list(range(prob.shape[1])))
    except ValueError:                                     # a class missing from this test set
        out['auc'] = float('nan')
    return out


def evaluate(yt, yl, pt, pl, rf=None):
    yt, yl = np.asarray(yt), np.asarray(yl)
    ht, hl = pt.argmax(1), pl.argmax(1)
    res = {'type': scores(yt, pt), 'loc': scores(yl, pl),
           'type_merged': dict(acc=accuracy_score(merge_lll(yt), merge_lll(ht)),
                               mcc=matthews_corrcoef(merge_lll(yt), merge_lll(ht))),
           'cm_type': confusion_matrix(yt, ht, labels=range(15)).tolist(),
           'cm_loc': confusion_matrix(yl, hl, labels=range(15)).tolist()}
    if rf is not None:
        shunt, res['by_rf'] = (yt >= 1) & (yt <= 11), []
        for lo, hi in RF_BINS:
            m = shunt & (rf > lo) & (rf <= hi)
            if m.any():
                res['by_rf'].append(dict(lo=lo, hi=hi, n=int(m.sum()),
                                         type_merged_acc=float(np.mean(merge_lll(ht[m]) == merge_lll(yt[m]))),
                                         loc_acc=float(np.mean(hl[m] == yl[m]))))
    return res


def report(res):
    lines = [f"{k:5s} acc {res[k]['acc']:.4f}  F1 {res[k]['f1']:.4f}  MCC {res[k]['mcc']:.4f}  "
             f"AUC {res[k]['auc']:.4f}" for k in ('type', 'loc')]
    lines.append(f"type, LLL/LLLG merged: acc {res['type_merged']['acc']:.4f}  MCC {res['type_merged']['mcc']:.4f}")
    for b in res.get('by_rf', []):
        lines.append(f"Rf {b['lo']:>5}-{b['hi']:<4} n={b['n']:4d}  type(merged) {b['type_merged_acc']:.3f}  "
                     f"loc {b['loc_acc']:.3f}")
    with np.printoptions(linewidth=200):
        lines += ['type confusion:', str(np.array(res['cm_type'])),
                  'location confusion:', str(np.array(res['cm_loc']))]
    return '\n'.join(lines)