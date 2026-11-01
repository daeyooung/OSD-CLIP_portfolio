import os
import sys
import numpy as np


def get_curve_online(known, novel, stypes=['Bas']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95


def metric_ood(x1, x2, stypes=['Bas'], verbose=True):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')

    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()

        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100. * tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[stype][mtype] = 100. * (-np.trapz(1. - fpr, tpr))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100. * (.5 * (tp[stype] / tp[stype][0] + 1. - fp[stype] / fp[stype][0]).max())
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUIN
        mtype = 'AUIN'
        denom = tp[stype] + fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype] / denom, [0.]])
        results[stype][mtype] = 100. * (-np.trapz(pin[pin_ind], tpr[pin_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')

        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0] - tp[stype] + fp[stype][0] - fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0] - fp[stype]) / denom, [.5]])
        results[stype][mtype] = 100. * (np.trapz(pout[pout_ind], 1. - fpr[pout_ind]))
        if verbose:
            print(' {val:6.3f}'.format(val=results[stype][mtype]), end='')
            print('')

    return results


def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR


def compute_overall_metrics(pred_k, pred_u, labels_k, num_classes, min_th=None, max_th=None, num_th=101):
    from sklearn.metrics import f1_score, accuracy_score
    # Inputs:
    # pred_k: numpy array (Nk, num_classes) - Softmax probabilities
    # pred_u: numpy array (Nu, num_classes) - Softmax probabilities
    # labels_k: numpy array (Nk,) - Ground truth labels for known samples
    # num_classes: int - Total number of known classes. Unknowns will be assigned label = num_classes
    
    # 1. Prepare Data
    # Known samples
    prob_k = np.max(pred_k, axis=1)
    pred_cls_k = np.argmax(pred_k, axis=1)
    
    # Unknown samples
    prob_u = np.max(pred_u, axis=1)
    # Initial prediction (will be overridden if prob < threshold)
    pred_cls_u = np.argmax(pred_u, axis=1) 
    
    # Aggregate
    all_probs = np.concatenate([prob_k, prob_u])
    all_initial_preds = np.concatenate([pred_cls_k, pred_cls_u])
    
    # GT Labels
    # Known: labels_k
    # Unknown: num_classes
    y_true = np.concatenate([labels_k, np.full(len(prob_u), num_classes)])
    
    # 2. Iterate Thresholds results in best F1
    # Sample 1000 thresholds from min to max probability
    min_p, max_p = np.min(all_probs), np.max(all_probs)
    thresholds = np.linspace(min_p, max_p, 101)
    
    max_f1 = -1.0
    best_th = 0.0
    best_acc = 0.0
    
    best_ur = 0.0
    
    # Define labels set to include all known classes + unknown class
    all_labels = np.arange(num_classes + 1)
    
    for th in thresholds:
        # Determine y_pred
        # If prob < th, predict 'Unknown' (num_classes)
        # Else, predict 'Initial Class' (0 to num_classes-1)
        
        y_pred = all_initial_preds.copy()
        y_pred[all_probs < th] = num_classes
        
        f1 = f1_score(y_true, y_pred, average='macro', labels=all_labels)
        
        if f1 > max_f1:
            max_f1 = f1
            best_th = th
            best_acc = accuracy_score(y_true, y_pred)
            
            # Calculate Recall for Unknown Samples (Best F1 threshold)
            # Correct Unknowns / Total Unknowns
            # y_true == num_classes is the unknown set
            is_unknown = (y_true == num_classes)
            num_unknown = np.sum(is_unknown)
            if num_unknown > 0:
                correct_unknown = np.sum((y_pred[is_unknown] == num_classes))
                best_ur = correct_unknown / num_unknown
            else:
                best_ur = 0.0
            
    return best_acc, max_f1, best_th, best_ur