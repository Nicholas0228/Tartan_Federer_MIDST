from sklearn.metrics import roc_curve

def get_tpr_at_fpr(true_membership: list, predictions: list, max_fpr=0.1) -> float:
    """Calculates the best True Positive Rate when the False Positive Rate is
    at most `max_fpr`.

    Args:
        true_membership (List): A list of values in {0,1} indicating the membership of a
            challenge point. 0: "non-member", 1: "member".
        predictions (List): A list of values in the range [0,1] indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more
            confident the predictor is about the hypothesis that the challenge point is
            a member.
        max_fpr (float, optional): Threshold on the FPR. Defaults to 0.1.

    Returns:
        float: The TPR @ `max_fpr` FPR.
    """
    fpr, tpr, _ = roc_curve(true_membership, predictions)

    return max(tpr[fpr < max_fpr])


import torch

def get_tpr_at_fpr_cuda(true_membership: torch.Tensor, predictions: torch.Tensor, max_fpr=0.1) -> float:
    """
    Calculates the best True Positive Rate (TPR) when the False Positive Rate (FPR) is
    at most `max_fpr` using PyTorch.

    Args:
        true_membership (torch.Tensor): A tensor of values in {0, 1} indicating the membership of a
            challenge point. 0: "non-member", 1: "member".
        predictions (torch.Tensor): A tensor of values in the range [0, 1] indicating the confidence
            that a challenge point is a member. The closer the value to 1, the more confident the predictor
            is about the hypothesis that the challenge point is a member.
        max_fpr (float, optional): Threshold on the FPR. Defaults to 0.1.

    Returns:
        float: The TPR @ `max_fpr` FPR.
    """
    # Ensure true_membership and predictions are tensors
    true_membership = true_membership.float()
    predictions = predictions.float()

    # Sort predictions in descending order and get the sorted indices
    sorted_indices = torch.argsort(predictions, descending=True)
    true_membership_sorted = true_membership[sorted_indices]
    
    # Calculate FPR and TPR
    positives = true_membership_sorted.sum()  # Total number of positive labels (members)
    negatives = true_membership_sorted.numel() - positives  # Total number of negative labels (non-members)
    
    # Initialize TPR and FPR lists
    tpr = torch.zeros_like(predictions)
    fpr = torch.zeros_like(predictions)
    
    # Calculate cumulative counts for TP and FP
    tp = torch.cumsum(true_membership_sorted, dim=0)
    fp = torch.cumsum(1 - true_membership_sorted, dim=0)
    
    # Calculate TPR and FPR
    tpr = tp / positives  # True Positive Rate (TPR)
    fpr = fp / negatives  # False Positive Rate (FPR)
    
    # Find the TPR at the max FPR
    best_tpr = tpr[fpr <= max_fpr].max()

    return best_tpr.item()
