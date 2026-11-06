| Task | Detector | C | AUC [95% CI] | TPR@5% FPR | TPR@1% FPR | Balanced acc. | macro-F1 [95% CI] | QWK |
|---|---|---|---|---|---|---|---|---|
| T1 Presence (all) | Stylometric | 0.25 | 0.789 [0.758, 0.821] | 0.369 | 0.102 | 0.700 | — | — |
| T1 Presence (all) | Char n-gram | 4 | 0.840 [0.814, 0.868] | 0.571 | 0.109 | 0.643 | — | — |
| T1 Presence (all) | Fused | 1 | 0.798 [0.769, 0.829] | 0.412 | 0.097 | 0.695 | — | — |
| T1 Presence (all) | BERT (frozen) | 0.25 | 0.812 [0.791, 0.835] | 0.448 | 0.177 | 0.717 | — | — |
| T1 Presence (effective edits only) | Stylometric | 0.25 | 0.819 [0.789, 0.852] | 0.405 | 0.166 | 0.738 | — | — |
| T1 Presence (effective edits only) | Char n-gram | 4 | 0.867 [0.841, 0.897] | 0.604 | 0.141 | 0.689 | — | — |
| T1 Presence (effective edits only) | Fused | 1 | 0.828 [0.800, 0.859] | 0.466 | 0.118 | 0.716 | — | — |
| T1 Presence (effective edits only) | BERT (frozen) | 0.25 | 0.844 [0.820, 0.871] | 0.489 | 0.197 | 0.761 | — | — |
| T2 Edit type (4-class) | Stylometric | 0.25 | — | — | — | — | 0.468 [0.441, 0.494] | — |
| T2 Edit type (4-class) | Char n-gram | 0.25 | — | — | — | — | 0.514 [0.483, 0.547] | — |
| T2 Edit type (4-class) | Fused | 1 | — | — | — | — | 0.475 [0.441, 0.507] | — |
| T2 Edit type (4-class) | BERT (frozen) | 0.25 | — | — | — | — | 0.490 [0.458, 0.520] | — |
| T2 Edit type (effective only) | Stylometric | 1 | — | — | — | — | 0.460 [0.430, 0.489] | — |
| T2 Edit type (effective only) | Char n-gram | 1 | — | — | — | — | 0.427 [0.391, 0.465] | — |
| T2 Edit type (effective only) | Fused | 4 | — | — | — | — | 0.478 [0.445, 0.511] | — |
| T2 Edit type (effective only) | BERT (frozen) | 0.25 | — | — | — | — | 0.499 [0.459, 0.535] | — |
| T3 Edit intensity (ordinal 0–4) | Stylometric | 4 | — | — | — | — | 0.379 [0.346, 0.409] | 0.404 |
| T3 Edit intensity (ordinal 0–4) | Char n-gram | 0.25 | — | — | — | — | 0.394 [0.371, 0.420] | 0.463 |
| T3 Edit intensity (ordinal 0–4) | Fused | 1 | — | — | — | — | 0.385 [0.350, 0.417] | 0.414 |
| T3 Edit intensity (ordinal 0–4) | BERT (frozen) | 0.25 | — | — | — | — | 0.381 [0.355, 0.407] | 0.441 |
| T4 Model attribution (4-class) | Stylometric | 16 | — | — | — | — | 0.551 [0.524, 0.575] | — |
| T4 Model attribution (4-class) | Char n-gram | 4 | — | — | — | — | 0.608 [0.576, 0.640] | — |
| T4 Model attribution (4-class) | Fused | 4 | — | — | — | — | 0.576 [0.547, 0.605] | — |
| T4 Model attribution (4-class) | BERT (frozen) | 0.25 | — | — | — | — | 0.504 [0.472, 0.533] | — |