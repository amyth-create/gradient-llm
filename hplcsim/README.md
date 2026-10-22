# hplcsim: real-compound sets from the HPLC Simulator library

A data root of its own, laid out like the project root so every loader finds
these sets by name (`sim.config.set_root`): a set's YAML, optimum, table and
results all live here, apart from the generated easy / medium / hard sets.

```
hplcsim/
  configs/external/   the HPLC Simulator compound library (43 measured compounds) and its provenance + licence
  configs/analytes/   hplcsim_beh.yaml, hplcsim_sb.yaml  (+ .optimum.json, + .import.json with the fit residuals)
  tables/             hplcsim_beh.csv, hplcsim_sb.csv    (22 080 grid methods each, same columns as tables/)
  results/            search plots and per-injection CSVs, landscapes, llm_transcripts/
```

| set         | source phase / modifier       | n  | optimum CRF | grid rows within 1 % of it |
|-------------|-------------------------------|----|-------------|----------------------------|
| hplcsim_beh | Waters Acquity BEH C18 / ACN  | 21 | 0.942       | 11                         |
| hplcsim_sb  | Agilent Zorbax SB-C18 / ACN   | 22 | 0.945       | 15                         |

Rebuild everything from the library (fit, optimum, table):

```bash
.venv/bin/python -m sim.import_hplcsim --table
.venv/bin/python -m sim.import_hplcsim --phase sb --solvent methanol --name hplcsim_sb_meoh --table   # a variant
.venv/bin/python -m sim.import_hplcsim --phase beh --ids 3 4 9 11 15 17 20 --name hplcsim_beh7 --table  # a subset
```

Run the optimizers on them exactly as on the generated sets; outputs land in
`hplcsim/results/`:

```bash
.venv/bin/python -m optimizers.run --set hplcsim_sb --methods random bo llm
.venv/bin/python -m optimizers.landscape --set hplcsim_beh --heatmaps-only
```

How the fit works, and how good it is, is described in the main README under
"Real-compound sets from the HPLC Simulator library". The library data are
CC BY-NC-SA 3.0 US (see `configs/external/README.md`).
