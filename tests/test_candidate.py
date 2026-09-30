import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import secactpy
from scipy import sparse

SOURCE = Path(__file__).parents[1] / "candidate" / "src" / "mymodel"


@dataclass(frozen=True)
class Prediction:
    ligand_scores: np.ndarray
    target_probability: np.ndarray
    up_probability: np.ndarray


def load_api(monkeypatch):
    candidate_api = types.ModuleType("ligand_activity.candidate_api")
    candidate_api.Prediction = Prediction
    monkeypatch.setitem(
        sys.modules, "ligand_activity", types.ModuleType("ligand_activity")
    )
    monkeypatch.setitem(sys.modules, "ligand_activity.candidate_api", candidate_api)
    spec = importlib.util.spec_from_file_location("mymodel.api", SOURCE / "api.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "mymodel.api", module)
    spec.loader.exec_module(module)
    return module


class Dataset:
    def __init__(self, tables, counts):
        self.tables, self._counts = tables, counts

    def table(self, name):
        return self.tables[name]

    def counts(self):
        return self._counts

    def ligand_universe(self, species):
        ligands = self.tables["ligands"]
        return ligands.ligand[ligands.species == species].tolist()

    def ligand_queries(self):
        queries = self.tables["queries"]
        return queries[queries.kind == "ligand"].reset_index(drop=True)

    def target_queries(self):
        queries = self.tables["queries"]
        return queries[queries.kind == "target"].reset_index(drop=True)


def synthetic():
    """SecAct knows Tnf and Il6; CytoSig knows TNFA, IL6 and GMCSF (Csf2).

    Each signature raises its own block of genes and lowers the next. Tnf is
    stimulated in qa and its signalling is lost in qb; Csf2 is stimulated in qc.
    """
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(300)]
    blocks = {"a": range(0, 30), "b": range(30, 60), "c": range(60, 90)}

    def signature(columns):
        values = rng.normal(0, 0.01, (len(genes), len(columns)))
        for j, block in enumerate(columns.values()):
            values[list(blocks[block]), j] += 1
            values[[i + 30 for i in blocks[block]], j] -= 0.5
        return pd.DataFrame(values, index=genes, columns=list(columns))

    signatures = {
        "secact": signature({"TNF": "a", "IL6": "b"}),
        "cytosig": signature({"TNFA": "b", "IL6": "c", "GMCSF": "c"}),
    }
    effect = {"qa": ("a", 4), "qb": ("a", 0.25), "qc": ("c", 4)}
    controls = [f"c{i}" for i in range(3)]
    mean = np.full(len(genes), 100.0)
    rows, links = [rng.poisson(mean) for _ in controls], []
    for query, (block, fold) in effect.items():
        response = np.ones(len(genes))
        response[list(blocks[block])] = fold
        for i in range(3):
            rows.append(rng.poisson(mean * response))
            links.append((query, f"{query}{i}", "treated"))
        links += [(query, c, "control") for c in controls]
    links.append(("qt", controls[0], "control"))
    queries = pd.DataFrame(
        {
            "query_id": ["qa", "qb", "qc", "qt"],
            "kind": ["ligand"] * 3 + ["target"],
            "species": "mouse",
            "direction": ["stimulation", "loss", "stimulation", "loss"],
            "ligands": [[], [], [], ["Tnf"]],
        }
    )
    samples = controls + [s for _, s, role in links if role == "treated"]
    tables = {
        "queries": queries,
        "query_samples": pd.DataFrame(links, columns=["query_id", "sample_id", "role"]),
        "samples": pd.DataFrame({"sample_id": samples}),
        "genes": pd.DataFrame({"gene": genes}),
        "ligands": pd.DataFrame(
            {"species": "mouse", "ligand": ["Tnf", "Il6", "Csf2", "Ccl2"]}
        ),
    }
    return Dataset(tables, sparse.csr_matrix(np.vstack(rows))), signatures, blocks


def test_predict_signs_loss_and_falls_back_to_cytosig(monkeypatch):
    api = load_api(monkeypatch)
    dataset, signatures, blocks = synthetic()
    monkeypatch.setattr(api, "load_signature", signatures.__getitem__)
    monkeypatch.setattr(
        api,
        "secact_activity_inference",
        lambda y, sig_matrix, **kwargs: secactpy.secact_activity_inference(
            y, sig_matrix=signatures[sig_matrix], is_group_sig=False, **kwargs
        ),
    )
    inputs = types.SimpleNamespace(dataset=dataset, pretrained=None)

    alone = api.predict(inputs, {"cytosig_fallback": False})
    both = api.predict(inputs, {"cytosig_fallback": True})

    ligands = ["Tnf", "Il6", "Csf2", "Ccl2"]
    for prediction in (alone, both):
        scores = pd.DataFrame(prediction.ligand_scores, columns=ligands)
        # Loss of Tnf signalling ranks Tnf first, like its stimulation.
        assert scores.loc[[0, 1]].idxmax(axis=1).eq("Tnf").all()
        assert (scores.Ccl2 == api.NO_SIGNATURE).all()
    # SecAct keeps Il6; CytoSig answers only for Csf2, which SecAct lacks.
    assert (alone.ligand_scores[:, 2] == api.NO_SIGNATURE).all()
    assert both.ligand_scores[2].argmax() == 2
    np.testing.assert_array_equal(alone.ligand_scores[:, :2], both.ligand_scores[:, :2])

    # Loss of Tnf lowers the genes its signature raises.
    up = both.up_probability[0]
    assert up[list(blocks["a"])].max() < 0.5 < up[list(blocks["b"])].min()
    assert both.target_probability[0, list(blocks["a"])].min() > 0.9
