from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix

# Used in _binary_clf_curve to determine whether two numbers are within 0.0001
# of one another.

def _isclose(a, b):
        
    return np.abs(a - b) < 0.0001

# Modified from https://github.com/scikit-learn/scikit-learn/blob/c957249/
# sklearn/metrics/ranking.py#L256

def _binary_clf_curve(y_true, y_score, pos_label=True, sample_weight=None):
    """
    Compute true and false positives per binary classification threshold.    

    Args:
        y_true: List of true targets of binary classification.
        y_score: List of estimated probabilities or decision function.
        pos_label: Label of positive class (default = True).
        sample_weight: List of sample weights (default = None).

    Returns:
        fps: List of false positive counts, the ith entry of which gives the
            number of negative samples assigned a score greater than or equal
            to the ith entry of thresholds.
        tps: List of true positive counts, the ith entry of which gives the
            number of positive samples greater than or equal to the ith entry
            of thresholds.
        thresholds: List of decreasing score values.
    """
    
    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # Make y_true a boolean vector.
    y_true = np.ravel([a==pos_label for a in y_true])

    # Sort scores and corresponding boolean values.
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # Extract the indices associated with distinct values (y_score typically
    # has many tied values). We also concatenate a value for the endpoint of
    # the curve. We use _isclore to avoid spurious repeated thresholds caused
    # by floating point rounding errors.
    distinct_value_indices = np.where(np.logical_not(_isclose(
        np.diff(y_score), 0)))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Accumulate true positives with decreasing threshold.
    tps = (y_true * weight).cumsum()[threshold_idxs]
    if sample_weight is not None:
        fps = weight.cumsum()[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

# Original scikit source: https://github.com/scikit-learn/scikit-learn/blob/
# c957249/sklearn/metrics/ranking.py#L417

# Original scikit documentation: scikit-learn.org/stable/auto_examples/
# model_selection/plot_roc_crossval.html

def get_aucec_curve_with_scores(y_true, y_scores, pos_label=True,
                                verbose=False):
    """
    Compute recall (R) and inspection ratio (IR) for each binary classification
    threshold. These values can be passed to sklearn.metrics.auc to find the
    area under the cost-effectiveness curve (AUCEC).    

    Args:
        y_true: List of true binary classification targets.
        y_score: List of estimated probabilities or decision function.
        pos_label: Label of positive class (default = True).
        verbose: Print debugging information (default = False).

    Returns:
        recall: List of recall values, the ith entry of which gives the recall
            corresonding to the ith inspection ratio.
        ir_list: Lost if inspection ratio values, the ith entry of which gives
            the inspection ratio corresponding to the ith threshold.
        thresholds: List of decreasing score values.
    """

    fps, tps, thresholds = _binary_clf_curve(y_true, y_scores, pos_label)
    
    if verbose:
	   print("true positives", tps)
	   print("false positives", fps)
	   print("thresholds", thresholds)

    # Get number of total positive instances.
    total_pos = sum([1 if a==pos_label else 0 for a in y_true])

    # Compute R and IR for each threshold using tps and fps from output of
    # _binary_clf_curve.
    recall_list = [ float(a) / total_pos for a in tps]
    ir_list = [float(tps[i] + fps[i]) / len(y_true) for i in range(len(tps))]

    # Add intersection point [0, 0].
    if ir_list[0] != 0:
	   ir_list = [0] + ir_list
	   recall_list = [0] + recall_list
           
    return ir_list, recall_list, thresholds

def get_aucec_curve(y_true, y_scores, pos_label=True, verbose=False):
    """
    Compute recall (R) and inspection ratio (IR) for each binary classification
    threshold. These values can be passed to sklearn.metrics.auc to find the
    area under the cost-effectiveness curve (AUCEC).

    Args:
        y_true: List of true binary classification targets.
        y_score: List of estimated probabilities or decision function.
        pos_label: Label of positive class (default = True).
        verbose: Print debugging information (default = False).

    Returns:
        recall: List of recall values, the ith entry of which gives the recall
            corresonding to the ith inspection ratio.
        ir_list: Lost if inspection ratio values, the ith entry of which gives
            the inspection ratio corresponding to the ith threshold.
    """
    
    fps, tps, thresholds = _binary_clf_curve(y_true, y_scores, pos_label)

    if verbose:
	   print("true positives", tps)
	   print("false positives", fps)
	   print("thresholds", thresholds)

    # Get number of total positive instances.
    total_pos = sum([1 if a==pos_label else 0 for a in y_true])

    # Compute R and IR for each threshold using tps and fps output from
    # _binary_clf_curve.
    recall_list = [ float(a) / total_pos for a in tps]
    ir_list = [float(tps[i] + fps[i]) / len(y_true) for i in range(len(tps))]

    # Add intersection point [0, 0].
    if ir_list[0] != 0:
	   ir_list = [0] + ir_list
	   recall_list = [0] + recall_list

    return ir_list, recall_list

def aucec_score(y_true, y_scores, pos_label=True, verbose=False):
    """
    Compute AUCEC score given list of true targets of classification and
    estimated probabilities. The AUCEC score is defined as the area under the
    cost-effectiveness curve [1].

    [1] James Walden, Jeff Stuckman, and Ricardo Scandariato. 2014. "Predicting
    Vulnerable Components: Software Metrics vs. Text Mining."2014 IEEE 25th
    International Symposium on Software Reliability Engineering, 23-33.

    Args:
        y_true: List of true binary classification targets.
        y_score: List of estimated probabilities or decision function.
        pos_label: Label of positive class (default = True).
        verbose: Print debugging information (default = False).

    Returns:
        AUCEC score for predictions.
    """
    
    ir_list, recall_list = get_aucec_curve(y_true, y_scores)
    return metrics.auc(ir_list, recall_list)

def aucec_50_score(y_true, y_scores, pos_label=True, verbose=False):
    """
    Compute AUCEC50 score for binary classifier given list of true targets of
    classification and estimated probabilities. The AUCEC50 score is defined as
    twice the area under the cost-effectiveness (CE) curve on 0 <= IR <= 0.5.

    In the context of software defect and vulnerability prediction, AUCEC50
    represents the most important IR cutoff. In general, code reviewers don't
    want to inspect more than 50% of the code base in order to find defects or
    vulnerabilities [1].

    [1] James Walden, Jeff Stuckman, and Ricardo Scandariato. 2014. "Predicting
    Vulnerable Components: Software Metrics vs. Text Mining."2014 IEEE 25th
    International Symposium on Software Reliability Engineering, 23-33.

    Args:
        y_true: List of true binary classification targets.
        y_score: List of estimated probabilities or decision function (either
            confidence values or probability estimates for the positive class).
        pos_label: Label of positive class (default = True).
        verbose: Print debugging information (default = False).

    Returns:
        AUCEC50 score for predictions.
    """
    
    ir_list, recall_list = get_aucec_curve(y_true, y_scores)
    ir_list_50 = [a for a in ir_list if a <= 0.5]
    recall_list_50 = recall_list[:(len(ir_list_50))]
    
    assert len(recall_list_50) == len(ir_list_50), \
           "the len of recall is " + str(len(recall_list_50))

    if ir_list_50[-1] != 0.5:
        # If there is no R value for IR = 0.5, add R value between IR values
        # immediately above and below IR = 0.5.
        next_ir = ir_list[len(ir_list_50)] #IR point above IR = 0.5.
        prev_ir = ir_list_50[-1] # IR point below IR = 0.5.
        next_recall = recall_list[len(recall_list_50)] # R point after R = 0.5.
        prev_recall = recall_list_50[-1] # R point below R = 0.5.

        # Calculate R value between IR points.
        slope = (next_recall - prev_recall) / (next_ir - prev_ir)
        r_50 = recall_list_50[-1] + slope * (.5 - ir_list_50[-1])

        # Append new point.
        ir_list_50.append(.5)
        recall_list_50.append(r_50)
        if (verbose): print("r50", r_50)
        
    return 2 * metrics.auc(ir_list_50, recall_list_50)

class My_CM:
    """
    Implementation of confusion matrix and common performance metrics used in
    software defect and vulnerability prediction.

    Attributes:
        tp: Number of true positives.
        tn: Number of true negatives.
        fp: Number of false positives.
        fn: Number of false negatives.
    """
    
    def __init__(self, array_cm):
        """
        Initialize confusion matrix from sklearn.metrics confusion matrix.
        """        
        
        self.tp = array_cm[1][1]
        self.tn = array_cm[0][0]
        self.fp = array_cm[0][1]
        self.fn = array_cm[1][0]

    def ir(self):

        """
        Compute inspection ratio (IR). Following [1], IR is defined as I / M,
        where I denotes the inspection (the number of code units covered by
        the prediction) and M the maximum possible inspection (which would be
        attained if every code unit was covered by the prediction). Formally,

            IR = I / M

        where

            I = TP + FP
            M = TP + TN + FP + FN.

        
        [1] James Walden, Jeff Stuckman, and Ricardo Scandariato. 2014.
        "Predicting Vulnerable Components: Software Metrics vs. Text Mining."
        2014 IEEE 25th International Symposium on Software Reliability
        Engineering, 23-33.
        
        Returns:
            Inspection ratio for predictions.
        """

        inspection = 1.0 * (self.tp + self.fp)
        max_inspection = self.tp + self.tn + self.fp + self.fn
        return inspection / max_inspection
    
    def f_score(self):
        """
        Compute F-score. The F-score is defined as the harmonic mean of recall
        and precision, and provides a balanced measure of those indicators.

        Returns:
            F-score for predictions.
        """

        p = self.precision()
        r = self.recall()
        return float(2 * (p * r) / (p + r)) if (p + r) != 0 else 0
    
    def accuracy(self):
        """
        Compute accuracy (ACC). Accuracy is defined as the proportion of
        correctly labeled positive and negative instances. In general, accuracy
        is not a good indicator of the performance of classifiers for software
        defect and vulnerability prediction. Formally,

            ACC = (TP + TN) / (TP + FN + FP + TN).

        Returns:
            Accuracy of predictions.
        """

        correct = self.tp + self.tn
        total = self.tp + self.tn + self.fp + self.tn
        return float(correct) / total
    
    def recall(self):
        """
        Compute recall (R). Recall is defined as the proportion of correctly
        identified positive instances out all true positive instances. Note
        that recall is often denoted PD in software defect and vulnerability
        prediction literature. Formally,
        
            R = TP / (TP + FN)

        Returns:
            Recall for predictions.
        """

        positive = self.tp + self.fn
        return float(self.tp) / positive if positive != 0 else 0
        
    def precision(self):
        """
        Compute precision. Precision is defined as the proportion of correctly
        identified positive instances out of all identified positive instances.

        Returns:
            Precision for predictions.
        """

        positive = self.tp + self.fp
        return float(self.tp) / positive if positive != 0 else 0

    def fp_rate(self):
        """
        Compute false positive rate (PF). The false positive rate is defined:
   
            PF = FP / (FP + TN).

        Returns:
            False positive rate for predictions.
        """
            
        return float(self.fp) / (self.fp + self.tn)

    def g_score(self):
        """
        Compute G-score. Following Menzies [1], the G-measure is defined as the
        harmonic mean of precision and specificity, where specificty is defined
        as one minus the false positive rate.

        [1] Fayola Peters, Tim Menzies, and Andrian Marcus. 2013. "Better
        Cross-Company Defect Prediction." IEEE International Working Conference
        on Mining Software Repositories, 409-18.
 
        Returns:
            G-score for predictions.
        """

        p = self.precision()
        s = 1 - self.fp_rate()
        return (2 * p * s) / (p + s)

    def __repr__(self):

        return "TP: "  + str(self.tp) + \
               " TN: " + str(self.tn) + \
               " FP: " + str(self.fp) + \
               " FN: " + str(self.fn)

def cm_for_ir_threshold(confidence, y_true, ir_threshold):
    """
    Create confusion matrix approximately satisfying a given inspection ratio
    threshold. This is accomplished by selecting a confidence threshold such
    that the number of positive instances as a proportion of the entire project
    is approximately equal to the given inspection ratio threshold.

    Args:
        confidence: List of estimated probabilities or decision function.
        y_true: List of true binary classification targers.
        ir_threshold: Inspection ratio threshold.

    Returns:
        My_CM object representing a confusion matrix approximately satisfying
        the specified inspection ratio threshold.
    """

    # Get inspection ratios and scores from AUCEC curve.
    ir_list, _, scores = get_aucec_curve_with_scores(y_true,confidence)

    # Find confidence threshold such that IR is approximately equal to the
    # specified IR threshold.
    confidence_threshold = scores[
        sum(np.array(np.array(ir_list) <= ir_threshold)) - 1
    ]

    # Make predictions based on new confidence threshold.
    y_pred = [x > confidence_threshold for x in confidence]

    # Return confusion matrix based on predictions.
    return My_CM(confusion_matrix(list(y_true), y_pred))
