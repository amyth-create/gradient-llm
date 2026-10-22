# External compound data

`hplcsim_compounds.json` is the compound library of the HPLC Simulator
(Boswell, Stoll, Carr et al., *J. Chem. Educ.* 2013, 90, 198-202,
https://doi.org/10.1021/ed300117b), copied from
https://github.com/pgboswell/hplc-simulator (`public/assets/simulator/compounds.js`,
header "Extracted from supplied v1.16 Globals.java"). It is redistributed here
unchanged in content under the project's licence, CC BY-NC-SA 3.0 US
(https://creativecommons.org/licenses/by-nc-sa/3.0/us/); keep this notice with it.

Two stationary phases, 43 real compounds, measured retention surfaces:

    log10 k = (w + m T) + (v + n T) phi        T in deg C, coefficients [m, w, n, v]

one coefficient row per organic modifier (acetonitrile, methanol).
`python -m sim.import_hplcsim` fits each surface to the simulator's own
`log_kw / S / dH` model and writes `configs/analytes/hplcsim_*.yaml`.
