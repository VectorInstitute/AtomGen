"""Utilities for data processing and evaluation."""

from typing import Any, Dict, Optional

import numpy as np
from scipy.special import expit, softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers.trainer_utils import EvalPrediction


def compute_metrics_smp(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute MAE for 20 regression labels for the SMP task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # Ensure predictions and labels are arrays, not tuples
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(label, tuple):
        label = label[0]

    # get Mean absolute error for each 20 pred and labels
    maes: Dict[str, Optional[float]] = {
        "rot_const_A": None,
        "rot_const_B": None,
        "rot_const_C": None,
        "dipole_moment": None,
        "isotropic_polarizability": None,
        "HOMO": None,
        "LUMO": None,
        "gap": None,
        "electronic_spatial_extent": None,
        "zero_point_vib_energy": None,
        "internal_energy_0K": None,
        "internal_energy_298.15K": None,
        "enthalpy_298.15K": None,
        "free_energy_298.15K": None,
        "heat_capacity_298.15K": None,
        "thermochem_internal_energy_0K": None,
        "thermochem_internal_energy_298.15K": None,
        "thermochem_enthalpy_298.15K": None,
        "thermochem_free_energy_298.15K": None,
        "thermochem_heat_capacity_298.15K": None,
    }
    for i in range(20):
        value = float(np.mean(np.abs(pred[:, i] - label[:, i])))
        maes[list(maes.keys())[i]] = value

    return maes


def compute_metrics_ppi(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute AUROC for the PIP task."""
    predictions = eval_pred.predictions
    label = eval_pred.label_ids

    # Ensure predictions and labels are arrays, not tuples
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(label, tuple):
        label = label[0]

    pred = expit(predictions > 0.5)

    # compute AUROC for each label
    for i in range(20):
        auroc = roc_auc_score(label[:, i], pred[:, i])

    return {"auroc": auroc}


def compute_metrics_res(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute accuracy for the RES task."""
    pred = softmax(eval_pred.predictions).argmax(axis=1)
    label = eval_pred.label_ids

    # compute accuracy

    acc = accuracy_score(label, pred)

    return {"accuracy": acc}


def compute_metrics_msp(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute AUROC for the MSP task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # compute AUROC for each label
    auroc = roc_auc_score(label, pred)

    return {"auroc": auroc}


def compute_metrics_lba(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute RMSE for the LBA task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # Ensure predictions and labels are arrays, not tuples
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(label, tuple):
        label = label[0]

    # compute RMSE for each label
    rmse = float(np.sqrt(np.mean((pred - label) ** 2)))
    global_pearson = float(pearsonr(pred.flatten(), label.flatten())[0])
    global_spearman = float(spearmanr(pred.flatten(), label.flatten())[0])

    return {
        "rmse": rmse,
        "global_pearson": global_pearson,
        "global_spearman": global_spearman,
    }


def compute_metrics_lep(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute AUROC for the LEP task."""
    pred = expit(eval_pred.predictions) > 0.5
    label = eval_pred.label_ids

    # compute AUROC for each label
    auroc = roc_auc_score(label, pred)

    return {"auroc": auroc}


def compute_metrics_psr(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute global spearman correlation for the PSR task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # Ensure predictions and labels are arrays, not tuples
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(label, tuple):
        label = label[0]

    # compute global spearman correlation
    global_spearman = float(spearmanr(pred.flatten(), label.flatten())[0])

    return {"global_spearman": global_spearman}


def compute_metrics_rsr(eval_pred: EvalPrediction) -> Dict[str, Any]:
    """Compute global spearman correlation for the RSR task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # Ensure predictions and labels are arrays, not tuples
    if isinstance(pred, tuple):
        pred = pred[0]
    if isinstance(label, tuple):
        label = label[0]

    # compute global spearman correlation
    global_spearman = float(spearmanr(pred.flatten(), label.flatten())[0])

    return {"global_spearman": global_spearman}
