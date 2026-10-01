"""SecAct ligand activity and ligand-target prediction.

Ligand queries run SecActpy's ridge-regression activity inference on each
query's pseudobulk log2 CPM change and rank ligands by activity z-score.
Target queries read the query ligands' signed signatures. Mouse genes and
ligands map to SecAct's human symbols by upper-casing, as SecActpy does.
"""

import json

import numpy as np
import pandas as pd
from ligand_activity.candidate_api import Prediction
from secactpy import load_signature, secact_activity_inference

DEFAULTS = {"cytosig_fallback": False}
# Ligands without a signature rank below every ligand with one.
NO_SIGNATURE = -1e6
# CytoSig names its signatures by cytokine; these map them to ligand symbols.
# Nitric oxide (NO) is not a protein ligand and stays unmapped.
CYTOSIG_LIGANDS = {
    "Activin A": ["INHBA"],
    "CD40L": ["CD40LG"],
    "GCSF": ["CSF3"],
    "GMCSF": ["CSF2"],
    # Type I interferons share IFNAR; human and mouse IFNA numbering differs.
    "IFN1": [f"IFNA{i}" for i in range(1, 22)]
    + ["IFNAB", "IFNB1", "IFNE", "IFNK", "IFNW1", "IFNZ"],
    "IFNL": ["IFNL1", "IFNL2", "IFNL3", "IFNL4"],
    "IL12": ["IL12A", "IL12B"],
    "IL36": ["IL36A", "IL36B", "IL36G"],
    "MCSF": ["CSF1"],
    "NO": [],
    "TNFA": ["TNF"],
    "TRAIL": ["TNFSF10"],
    "TWEAK": ["TNFSF12"],
}


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("reference", {}).get("parameters", {})
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown SecAct parameters: {sorted(unknown)}")
    return DEFAULTS | values


def signatures(config):
    """SecActpy's signature names, each with the ligand symbols it answers for.

    Without the CytoSig fallback, only SecAct answers. With it, CytoSig answers
    for ligands that SecAct lacks.
    """
    secact = load_signature("secact")
    chosen = [("secact", secact, {ligand: ligand for ligand in secact.columns})]
    if config["cytosig_fallback"]:
        cytosig = load_signature("cytosig")
        ligands = {
            ligand: name
            for name in cytosig.columns
            for ligand in CYTOSIG_LIGANDS.get(name, [name])
            if ligand not in secact.columns
        }
        chosen.append(("cytosig", cytosig, ligands))
    return chosen


def sign(queries):
    """+1 for stimulation and -1 for loss of signalling."""
    return np.where(queries.direction.to_numpy() == "loss", -1.0, 1.0)


def change(dataset, queries, genes):
    """Each ligand query's mean treated minus mean control log2 CPM.

    This is SecActpy's pseudobulk change: counts to CPM, log2(x + 1), and the
    difference from the control mean. Upper-cased duplicate symbols keep the
    most expressed gene, as in SecActpy.
    """
    samples = dataset.table("samples")
    links = dataset.table("query_samples")
    counts = dataset.counts()
    library = np.maximum(np.asarray(counts.sum(axis=1)).ravel(), 1)
    symbols = dataset.table("genes").gene.str.upper()
    total = pd.Series(np.asarray(counts.sum(axis=0)).ravel())
    keep = (
        total[symbols.isin(genes).to_numpy()]
        .groupby(symbols[symbols.isin(genes)].to_numpy())
        .idxmax()
    )
    counts = counts[:, keep.to_numpy()]
    row = pd.Series(np.arange(len(samples)), index=samples.sample_id)
    result = np.zeros((len(keep), len(queries)))
    for j, query_id in enumerate(queries.query_id):
        members = links[links.query_id == query_id]
        treated = row[members.sample_id[members.role == "treated"]].to_numpy()
        control = row[members.sample_id[members.role == "control"]].to_numpy()
        rows = np.concatenate([treated, control])
        log_cpm = np.log2(counts[rows].toarray() * (1e6 / library[rows])[:, None] + 1)
        result[:, j] = log_cpm[: len(treated)].mean(axis=0) - log_cpm[
            len(treated) :
        ].mean(axis=0)
    return pd.DataFrame(result, index=keep.index, columns=queries.query_id.to_numpy())


def ligand_scores(dataset, queries, universe, chosen):
    symbols = pd.Index(universe).str.upper()
    result = np.full((len(queries), len(universe)), NO_SIGNATURE)
    if not len(queries):
        return result
    genes = set().union(*(signature.index for _, signature, _ in chosen))
    y = change(dataset, queries, genes)
    direction = sign(queries)
    for name, _, ligands in chosen:
        # SecActpy renames a single column, so results are read by position.
        z = secact_activity_inference(
            y, is_differential=True, sig_matrix=name, verbose=False
        )["zscore"]
        for j, symbol in enumerate(symbols):
            if symbol in ligands:
                result[:, j] = direction * z.loc[ligands[symbol]].to_numpy()
    return result


def target_predictions(dataset, queries, chosen):
    """Signature weight scaled per ligand to [-1, 1]; strongest ligand wins.

    The signed scaled weight stands in for the gene's log2 fold change.
    """
    symbols = dataset.table("genes").gene.str.upper()
    weights = {}
    for _, signature, ligands in chosen:
        for ligand, column in ligands.items():
            if ligand not in weights:
                values = signature[column].reindex(symbols).fillna(0).to_numpy()
                peak = np.abs(values).max()
                weights[ligand] = values / peak if peak > 0 else values
    scaled = np.zeros((len(queries), len(symbols)))
    for i, ligands in enumerate(queries.ligands):
        for ligand in ligands:
            values = weights.get(ligand.upper())
            if values is not None:
                stronger = np.abs(values) > np.abs(scaled[i])
                scaled[i, stronger] = values[stronger]
    return scaled * sign(queries)[:, None]


def predict(inputs, config):
    dataset = inputs.dataset
    species = dataset.table("queries").species.unique()
    if len(species) != 1:
        raise ValueError("a held-out dataset has queries of one species")
    chosen = signatures(config)
    return Prediction(
        ligand_scores(
            dataset,
            dataset.ligand_queries(),
            dataset.ligand_universe(species[0]),
            chosen,
        ),
        target_predictions(dataset, dataset.target_queries(), chosen),
    )
