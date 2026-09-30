# SecAct (SecActpy)

[SecAct](https://doi.org/10.1038/s41592-026-03172-0) (Ru, Jiang et al.,
*Nature Methods*, 2026) infers the signalling activity of 1,170 human secreted
proteins by ridge regression of an expression change on their signed response
signatures. The signatures were learned from co-variation around each protein
across 1,258 cancer spatial transcriptomics samples, not from ligand-treatment
experiments. This reference wraps
[SecActpy](https://github.com/data2intelligence/SecActpy), the authors' Python
port, for the `ligand_activity` task. The signatures ship inside the pinned
package, so the candidate needs no training or pretrained artifact.

## Presets

- `secact` uses the SecAct signatures alone, which is upstream's default.
- `secact_cytosig` answers for ligands that SecAct lacks with SecActpy's
  bundled [CytoSig](https://doi.org/10.1038/s41592-021-01274-5) signatures.
  SecAct misses several classic immune cytokines, such as IL-2, IL-3, IL-4,
  IL-12, IL-13, IL-17A and type I interferons, probably because they are rarely
  detectable in tumour tissue. CytoSig names signatures by cytokine; the
  wrapper maps them to ligand genes, for example `IFN1` to every type I
  interferon and `IL12` to both subunits. CytoSig was fitted on bulk treatment
  profiles, including those behind the task's CytoSig training contribution. It
  predates both validation atlases.

## Adaptation

- **Species.** Signatures are human. Mouse genes and ligands map to them by
  upper-cased symbol, as SecActpy does for its own inputs; mouse-only genes are
  lost. SecAct covers about 45% of each species' ligand universe and 7,919 genes.
- **Ligand queries.** Each query's change is its mean treated minus mean
  control log2(CPM + 1), which is SecActpy's own pseudobulk contrast, against
  all of the receiver's shown controls. All queries of a dataset run through one
  upstream call with default parameters, including signature grouping and 1,000
  permutations. Ligands rank by activity z-score. Ligands without a signature
  rank below every ligand with one. SecAct scores secreted proteins, not
  receptor usage, so receptor expression is not used.
- **Target queries.** A gene's target probability is the absolute signature
  weight of the query ligand, scaled by that ligand's largest absolute weight;
  its up-regulation probability maps the signed scaled weight to [0, 1]. With
  several ligands, the ligand with the largest absolute weight decides. Genes
  outside the signatures get probability 0 and direction 0.5.
- **Loss of signalling.** Loss conditions negate ligand z-scores and flip target
  direction.
- SecActpy caps numpy below 2, which has no Python 3.13 wheels, and candidate
  environments cannot override a dependency's metadata. The candidate therefore
  pins the `secactpy-wheel-v0.3.1` release asset of this repository, which
  [`scripts/build_secactpy_wheel.py`](scripts/build_secactpy_wheel.py) builds
  from the upstream PyPI wheel by removing only that cap. SecActpy uses no numpy
  API that version 2 removed.

The method is deterministic and uses no training data, so the candidate is
predict-only and the reference does not run the task's train route.

## Limitations

Spatial co-variation does not separate a cytokine from the secreted proteins
it induces. On the Immune Dictionary training slice, IFN-γ responses are
attributed to CXCL10, CXCL11 and CCL8 ahead of IFN-γ itself. Target signatures
describe tumour tissue rather than any receiver cell type, so the method cannot
predict cell-type-specific targets.

On validation, SecAct ranks ligands less well than NicheNet and than the
generic responder, and it earns almost no unsigned target credit. Its signed
signatures earn some direction credit, which NicheNet cannot, and the CytoSig
fallback improves both components.
