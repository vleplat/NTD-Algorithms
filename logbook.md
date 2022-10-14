# Logs: who did what and when

## Jeremy, 31/08/22
- Reconnect with the material, start logbook
- Prep scripts for synthetic comparisons
  - script for acceleration vs no acceleration
  - updating shootout to plot median cvg plots with deviation
- Visio with Valentin
- q: delete latex folder? Up to date is on Overleaf

## Jeremy, 01/09/22
- Still working on script for median plot
  - median on iterations with pandas (in shootout, todo test)
  - 0.2 and 0.8 quantiles with pandas
  - put error bars in px.lines
  - temporal interpolation for time plots (in shootout, todo test)
- Script for ex vs noex
- added sparsity input with one scalar --> only in KL, do it in HALS l2.
- Now tracking sparsity to check it is not over the top in experiments. Good values empirically in [1,10]. 100 makes almost all zero.
- Running acc vs no acc (with and without extrapolation) and ext vs no ext (with various sparsity levels), results stored, plots TODO (did acc vs no acc)
- Noticed acceleration not done for core update in KL --> did it; now should be everywhere.
- Observation: acceleration helps, espcially good with e.g. 20 iter inner max, helps prune useless iterations in later phases of the algorithm (almost no impact on first iters), see png.
- corrected small bug in cost, l2 missing 1/2 and square.
- for simulations, dense factors and sparse core ?? TODO: both settings (looped over)
- Checking extrapolation: alpha max is always 0.99 ??
- Change alpha_reduce from 0.8 to 0.2 for testing (because max increased)
- BUG TO INVESTIGATE: no sparsity on core when extrapolation is OFF (wtf???) whatever l1weight --> in fact it works but the values for the sparsity are different than with extrapolation (wt wtf??)
- The losses are a mess in one-loop-beta-div, TODO: simplify

## Jeremy, 02/09/22

- TODO an xp l2 vs nol2 ?? Since with beta=1 the update rule changes?
- changed l2 weights to be part of shootout input for storing it
- Extrapolation test/acceleration test: visible impact if size large enough? testing 100x40 instead of 40x40 for acc vs noacc
- Remark: acceleration has a small additional cost (compute norm of deltaX) but this seems to be measurable in the experiments.
- Tracking number of inner loops with prints. Seeing that factors stop earlier than core. Also factors always stop after 1 inner at first iter. Correcting by first outer loop no acceleration.
- To tune down complexity, use l1 norm instead of l2? just a sum no products!
- Added sqrt(acc_delta) to core so that inner stopping is scaled (empirical)
- Acceleration should work properly, with little additional cost (but not 0). Launched script again with new code, extrapolation off and small dimensions to check.
- Bug? why do several run with no acceleration (e.g. 5 inner iters) don't superpose (init should be seeded, the algorithm should run exactly the same?) Maybe compare raw results and not median to sanity check --> interpolation was bugged, corrected
- Increased number of iterations to reach convergence --> 1500
- Run mu_ntd/xps/Results/xp_acc_noacc_02-09-22 is interesting --> acceleration helps a little bit when it is mild, but we need these inner iters to convergence nicely (especially clear on iter graph). Raises question below:
- TODO: test just inner iter, with larger outer iters for smaller inner iters, acceleration off. Used linear scaling of outer iters with inner iters, seems to keep runtime in check ~1min30 for a run.
- Did adaptive grid for plots: todo debut when seeds>1

 ## Jeremy, 11/10/22

 - Committing latest changes and pushing
 - Running xp_inner to get back on the project, checking new plots with updated shootout
 - Restoring old acceleration (squared l2 norm of updates, to match other works)
 - Inner impact --> change from inner grid to fixed inner iters and delta grid
 - Delta choice seems to not help converge faster, in particular compared to fixing to e.g. 10 inner iterations. In particular delta is impacts the core and the factors differently. Doing a few more experiments in this direction.
 - TODO: investigate extrapolation changing sparsity values, probably bug.

## Jeremy, 12/10/22

- Improving acceleration experiment by checking both inner iters and acceleration (in particular small nb of inner iters/high delta)
- changing seed to have hash("sNTD")
- Removed sparse core always, since small dimensions may lead to 0 core with reasonable probability
- I would say setting 50 inner iters is a good compromise. No need for acceleration. Acceleration is erratic because G update is different.
- Deprecated acc vs no acc, included in inner iters
- Extrapolation TODO: test with other dims and more seeds
- TODO: test 50 iterations, extrapol and no extrapol? No need to tweak inner iters indefinitely. Also sparse data vs no sparse data.
- TODO: plot sparsity levels as boxplots?

## Jeremy, 13/10/22

- Note: Maybe we should fix default inner iters differently from core and factors, but OK let's simplify and keep 50.
- Trying to spot the bug in extrapolate; found that beta=2 l1 update is broken in litterature, but that is not the source of the bug. Removed gamma(beta) for beta in [1,2] as it is 1.
- IMPORTANT: EXTRAPOLATION MODIFICATIONS WRT ANDY:
  - alphamax upper bound = 2, not 0.99.
  - Computing cost on primary sequence since no speed up otherwise.
  - Not taking the second sequence as output (only changes last iter)
  - This yields a clearer extrapolation procedure.
- Rerunning extrapolation vs no extrapolation after this update. Hopefully not everything will break ahah... Its fine !
- TODO: return loss without l1 and l2 to compare

## Jeremy, 14/10/22

- What level of regularization?
- Showing facet plots for lambdas
- Correcting bug in updates that made algorithm not do anything.
- Adding sparsity tracking output
- Extrapolation bug corrected :) :) Sparsity coefficients behave the same in both ex and no ex settings :)
- Everything is fine with sparsity :)
- Note: discussing theory behind several l1 pens is not trivial.