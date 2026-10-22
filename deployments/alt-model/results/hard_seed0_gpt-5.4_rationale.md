# gpt-5.4 on the hard set, seed 0: the model's rationale for each injection

Taken verbatim from `deployments/alt-model/results/llm_transcripts/hard_seed0.jsonl`, the campaign
plotted in `hard_seed0_gpt-5.4_vs_gpt-6-astra.png`. The method line is what was actually run after
snapping to the grid; CRF is its score and best the campaign's best so far.

## Injection 1: phi0 0.60, phi1 0.95, tG 20 min, T 40 C -> CRF 0.183 (best 0.183)

Start with a broad mid-temperature scouting gradient to reveal the full elution window and avoid truncation while giving enough slope for informative spacing across 10 unknowns. This first run should show whether peaks are unretained, held to the final hold, or distributed within the gradient so we can tighten phi and tG next.

## Injection 2: phi0 0.05, phi1 0.95, tG 40 min, T 25 C -> CRF 0.190 (best 0.190)

The first run started far too strong, leaving 9 peaks essentially unretained at the dead time, so the priority is to lower phi0 to the minimum and use the shallowest, coolest broad gradient to maximize early retention and spread. Keeping phi1 high avoids trapping the last compound while we learn whether the critical cluster can be pulled into the gradient window.

## Injection 3: phi0 0.05, phi1 0.60, tG 20 min, T 25 C -> CRF 0.369 (best 0.369)

All compounds eluted and only one pair is critical, but the run is far too long; most peaks appear to leave within roughly phi 0.12–0.26, so the next step is to trim the gradient window hard around the observed elution window and shorten tG to keep all peaks in-gradient while cutting the late time penalty. This should test whether we can preserve the near-resolved pair while pulling the last peak much earlier.

## Injection 4: phi0 0.05, phi1 0.40, tG 20 min, T 25 C -> CRF -0.744 (best 0.369, truncated: 1 peak(s) missing)

The elution window is actually narrow and low in organic, with the last peak still coming well after the gradient, so lowering phi1 should pull the whole separation into the active gradient region and improve efficiency without sacrificing the strong early retention from low phi0. This also probes whether the critical 3/4 pair benefits from a tighter phi window rather than a shallower overall slope.

## Injection 5: phi0 0.05, phi1 0.45, tG 20 min, T 25 C -> CRF -0.755 (best 0.369, truncated: 1 peak(s) missing)

Dropping phi1 to 0.40 improved the critical pair to R 1.05 but trapped the most retained compound, so the next move is a small step up in phi1 to recover full elution while preserving that gain in selectivity. This should tell us whether the optimum lies near the edge where the last compound just comes off and the 3/4 pair is still improved.

## Injection 6: phi0 0.05, phi1 0.50, tG 20 min, T 25 C -> CRF -0.112 (best 0.369)

Phi1=0.45 still truncates, so increase phi1 one more step to bring the last compound off while staying much tighter than 0.60, where full elution occurred. This bracket should locate the minimum phi1 that gives 10/10 peaks and ideally keep the critical pair better resolved than at phi1=0.60.

## Injection 7: phi0 0.05, phi1 0.60, tG 20 min, T 40 C -> CRF 0.546 (best 0.546)

The critical 3/4 pair barely responds beneficially to slope/window changes, so temperature is the next lever to test for selectivity while also shortening the run substantially. Reusing the current best fully eluting composition at higher T should show whether that close pair can be reordered or separated better without the heavy time penalty.

## Injection 8: phi0 0.05, phi1 0.60, tG 12 min, T 40 C -> CRF 0.877 (best 0.877)

Now that all pairs are fully resolved, the objective is purely to reduce t_last; the fastest way is to shorten tG while keeping the successful phi window and temperature fixed. This run probes how much gradient time can be removed before any pair drops below R=1.5.

## Injection 9: phi0 0.05, phi1 0.60, tG 10 min, T 40 C -> CRF 0.965 (best 0.965)

All pairs still have ample margin at tG=12, so shorten the gradient another step to push the last peak toward the 10 min target. Keeping the same window and temperature isolates the speed/resolution tradeoff and should reveal whether we can gain more CRF without losing full-credit resolution.

## Injection 10: phi0 0.05, phi1 0.60, tG 8 min, T 40 C -> CRF 0.965 (best 0.965)

At tG=10 the weakest pair still has R=1.66, so there is room to steepen once more and likely bring the last peak under 10 min for a perfect score. This is the direct next step on the speed frontier with the proven successful phi window and temperature.

## Injection 11: phi0 0.05, phi1 0.60, tG 8 min, T 35 C -> CRF 0.919 (best 0.965)

tG=8 gets the last peak under 10 min but loses score on the 6/7 pair, so the next lever is a small temperature drop to recover selectivity with minimal impact on time. This directly tests whether that near-critical pair can be pushed back above R=1.5 while keeping t_last at or below 10 min.

## Injection 12: phi0 0.05, phi1 0.60, tG 10 min, T 45 C -> CRF 0.916 (best 0.965)

The best current score is tied between tG 10 and 8 at 40 C, and the 8 min run became under-resolved; a modest temperature increase at tG 10 may pull the last peak below 10 min while retaining the comfortable resolution margin seen at 40 C. This is the most promising route to a true CRF 1.0 without changing the proven gradient window.

## Injection 13: phi0 0.05, phi1 0.55, tG 10 min, T 40 C -> CRF 0.900 (best 0.965)

Temperature changes worsened the critical cluster, so return to 40 C and test a slightly narrower gradient window, which often improves selectivity for close pairs while keeping the run near the 10 min target. Lowering phi1 from 0.60 to 0.55 should modestly strengthen retention around the crowded mid-region without risking truncation.

## Injection 14: phi0 0.05, phi1 0.65, tG 10 min, T 40 C -> CRF 0.992 (best 0.992)

The narrower window improved resolution but made the run too long, while the best run at phi1=0.60 is only 0.71 min above the 10 min target; a small increase in phi1 may shave enough time to reach t_last <=10 without sacrificing the still-healthy resolution margin. This is the cleanest way to search for a perfect 1.0 around the current optimum.

## Injection 15: phi0 0.05, phi1 0.65, tG 12 min, T 45 C -> CRF 0.891 (best 0.992)

The current best is extremely close to perfect, with only the 6/7 pair slightly below full-credit resolution; a modest increase in tG should raise that pair above R=1.5, and the higher temperature can offset the added gradient time to keep the last peak at or under 10 min. This balances the two score terms most directly in pursuit of CRF 1.0.

## Injection 16: phi0 0.05, phi1 0.65, tG 12 min, T 40 C -> CRF 0.921 (best 0.992)

At phi1=0.65 and 10 min, the only remaining gap to CRF 1.0 is the 6/7 pair at R=1.39; increasing tG one step at the same temperature is the most direct way to raise that pair above 1.5. Although t_last will rise, a perfect resolution term may outweigh a small time penalty if the last peak stays close to 10 min.

## Injection 17: phi0 0.05, phi1 0.70, tG 10 min, T 40 C -> CRF 0.975 (best 0.992)

At tG=10, raising phi1 from 0.60 to 0.65 nearly hit the ideal balance, giving t_last just under 10 min while only slightly under-resolving 6/7; one more small increase in phi1 may shift that critical pair enough to reach full credit without reintroducing a time penalty. This is the most targeted remaining search around the current optimum.

## Injection 18: phi0 0.05, phi1 0.65, tG 10 min, T 35 C -> CRF 0.913 (best 0.992)

Phi1=0.65 at 40 C is the best point so far, but the 6/7 pair is just shy of full credit; a small drop in temperature may increase selectivity enough to push that pair above R=1.5 while only modestly increasing retention from the current t_last 9.96 min. This is a focused attempt to convert the near-perfect run into CRF 1.0.

## Injection 19: phi0 0.05, phi1 0.65, tG 8 min, T 45 C -> CRF 0.890 (best 0.992)

Since lowering temperature hurt selectivity and 8 min at 40 C under-resolved only slightly, test the opposite corner: a shorter gradient with higher temperature, which may shift the critical cluster differently while preserving t_last comfortably below 10 min. This is a remaining distinct way to seek a full-credit resolution pattern near the current optimum.

## Injection 20: phi0 0.05, phi1 0.65, tG 8 min, T 50 C -> CRF 0.946 (best 0.992)

The close pair is highly temperature-sensitive, and 35/45 C both shifted selectivity strongly; testing 50 C at the fast 8 min, phi1 0.65 condition probes whether a different thermal selectivity regime separates the critical pair while keeping the run very short. With only a few injections left, this is a distinct unexplored direction around the best composition.

## Injection 21: phi0 0.05, phi1 0.65, tG 8 min, T 30 C -> CRF 0.815 (best 0.992)

The critical cluster changes identity with temperature, so a colder setting at the fast 8 min, phi1 0.65 condition is still worth probing as another distinct selectivity regime. If 30 C restores separation while keeping t_last under 10 min, it could exceed the current near-best score.

## Injection 22: phi0 0.05, phi1 0.75, tG 10 min, T 40 C -> CRF 0.962 (best 0.992)

The best point remains 0.05/0.65/10/40, and increasing phi1 to 0.70 shortened the run but under-resolved the same 6/7 pair; one final phi1 step tests whether that pair’s selectivity is nonmonotonic with composition while the run stays comfortably under 10 min. With few injections left, this is the last nearby composition variable not yet sampled on the best tG/T setting.

## Injection 23: phi0 0.05, phi1 0.65, tG 10 min, T 50 C -> CRF 0.927 (best 0.992)

The critical 6/7 pair is very temperature-sensitive and has not been tested at 50 C on the best tG=10, phi1=0.65 setting; this is a distinct remaining chance that thermal selectivity shifts the pair above R=1.5 while keeping t_last safely under 10 min. With only three injections left, it is the highest-upside unexplored point near the optimum.

## Injection 24: phi0 0.05, phi1 0.80, tG 10 min, T 40 C -> CRF 0.950 (best 0.992)

The best run at phi1=0.65 is only limited by a single pair at R=1.39, and the composition trend from 0.65 to 0.75 is not disastrous; one last phi1 increase tests whether that pair’s selectivity shows a local maximum farther up the gradient while t_last remains safely below 10 min. This is the last nearby unexplored composition point on the strongest tG/T setting.

## Injection 25: phi0 0.10, phi1 0.65, tG 10 min, T 40 C -> CRF 1.000 (best 1.000)

The best region is clearly around phi1 0.65, tG 10, and 40 C, so the last distinct nearby test is a small increase in phi0 to see whether trimming the very weakly retained start changes the crowded mid-cluster selectivity enough to lift the limiting pair above R=1.5 while keeping t_last under 10 min. This is the closest unexplored perturbation around the campaign optimum.
