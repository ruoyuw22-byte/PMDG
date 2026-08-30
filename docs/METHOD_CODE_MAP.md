# Method-to-code map

| Manuscript component | Primary implementation | Main outputs |
| --- | --- | --- |
| DICOM standardization | `src/dagl_mil/data/preprocessing.py` | normalized image, breast mask, laterality transform |
| Patch bag construction | `src/dagl_mil/data/patches.py` | patches, coordinates, coverage, anatomical descriptors |
| Breast-level sample | `src/dagl_mil/data/dataset.py` | ipsilateral CC/MLO ragged bags and two labels |
| Mammo-CLIP encoder | `src/dagl_mil/models/mammoclip/backbone.py` | 352-D patch features and optional feature pyramid |
| Anatomical feature fusion | `AnatomicalFeatureFusion` in `models/dagl.py` | anatomy-aware candidate tokens |
| Prototype-guided grouping | `PrototypeMemory` in `models/dagl.py` | normal/mass/calcification prototype responses |
| Sparse dynamic graph | `DynamicAnatomyGraph` in `models/dagl.py` | graph-refined candidate tokens and edge weights |
| Cross-view consistency | `CrossViewConsistency` in `models/dagl.py` | CC↔MLO soft correspondence and consistency loss |
| Top-K MIL aggregation | `TopKClassAggregator` in `models/dagl.py` | breast logits and selected candidate masks |
| Joint objective | `DAGLMIL.forward` in `models/dagl.py` | classification, prototype, diversity, graph, cross-view and sparsity losses |
| Training loop | `src/dagl_mil/engine/trainer.py` | `best.pt`, `last.pt`, history and provenance |
| Fixed inference pass | `src/dagl_mil/engine/inference.py` | immutable breast/view prediction records |
| Heatmap reconstruction | `src/dagl_mil/evaluation/localization.py` | full-resolution response maps and boxes |
| Classification metrics | `src/dagl_mil/evaluation/classification.py` | ROC-AUC, PR-AUC and threshold metrics |
| Bootstrap inference | `src/dagl_mil/evaluation/bootstrap.py` | confidence intervals and paired differences |
| Baselines | `src/dagl_mil/models/baselines.py` | ABMIL, Set Transformer, FPN-SetTrans, PatchGCN-MIL, Cross-view Transformer-MIL |
| Ablations | `configs/ablations/*.yaml` | seven manuscript-aligned module removals |

The localization boxes in VinDr-Mammo are never exposed by training or
validation datasets. They enter only the fixed test inference export for
post-hoc localization evaluation.

