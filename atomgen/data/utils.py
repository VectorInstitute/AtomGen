import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import expit, softmax
from scipy.stats import pearsonr, spearmanr



def compute_metrics_smp(eval_pred):
    """Compute MAE for 20 regression labels for the SMP task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # get Mean absolute error for each 20 pred and labels
    maes = {
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
        value = np.mean(np.abs(pred[:, i] - label[:, i]))
        maes[list(maes.keys())[i]] = value

    return maes

def compute_metrics_ppi(eval_pred):
    """Compute AUROC for the PIP task."""
    pred = expit(eval_pred.predictions > 0.5)
    label = eval_pred.label_ids

    # compute AUROC for each label
    for i in range(20):
        auroc = roc_auc_score(label[:, i], pred[:, i])

    return {"auroc": auroc}

def compute_metrics_res(eval_pred):
    """Compute accuracy for the RES task."""
    pred = softmax(eval_pred.predictions).argmax(axis=1)
    label = eval_pred.label_ids

    # compute accuracy
        
    acc = accuracy_score(label, pred)

    return {"accuracy": acc}

def compute_metrics_msp(eval_pred):
    """Compute AUROC for the MSP task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # compute AUROC for each label
    auroc = roc_auc_score(label, pred)

    return {"auroc": auroc}

def compute_metrics_lba(eval_pred):
    """Compute RMSE for the LBA task."""

    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # compute RMSE for each label
    rmse = np.sqrt(np.mean((pred - label) ** 2))
    global_pearson = pearsonr(pred.flatten(), label.flatten())[0]
    global_spearman = spearmanr(pred.flatten(), label.flatten())[0]

    return {"rmse": rmse, "global_pearson": global_pearson, "global_spearman": global_spearman}

def compute_metrics_lep(eval_pred):
    """Compute AUROC for the LEP task."""
    pred = expit(eval_pred.predictions) > 0.5
    label = eval_pred.label_ids

    # compute AUROC for each label
    auroc = roc_auc_score(label, pred)

    return {"auroc": auroc}

def compute_metrics_psr(eval_pred):
    """Compute global spearman correlation for the PSR task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # compute global spearman correlation
    global_spearman = spearmanr(pred.flatten(), label.flatten())[0]

    return {"global_spearman": global_spearman}

def compute_metrics_rsr(eval_pred):
    """ Compute global spearman correlation for the RSR task."""
    pred = eval_pred.predictions
    label = eval_pred.label_ids

    # compute global spearman correlation
    global_spearman = spearmanr(pred.flatten(), label.flatten())[0]

    return {"global_spearman": global_spearman}