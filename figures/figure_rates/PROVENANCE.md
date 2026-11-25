# How the chemotaxis data of the paper was produced

This document records how the data files in `data/` were made, as far as it could be reconstructed
from the author's cluster account (scripts, notebooks, per-job parameter files and a manifest of the
raw output). The scripts and notebooks mentioned here are **not part of this repository**; their
location is given so that they can be found or copied in later. Statements are marked
*(inferred)* where they follow from matching parameters and not from a record.

`reproduce_chemotaxis.jl` recomputes the two data files of the main chemotaxis figure
(`2023-02-17_mi_trajectories.csv` and `input_traj_summary.csv`) and is documented in
[`README.md`](README.md). The other data files were produced as described below.

## Raw data archive

The raw HDF5 files of the three runs behind the figures (`2023-02-17`, `2023-01-15`, `2023-01-11`), the
submission and run scripts, the job configurations and the aggregation notebooks are archived at
[doi:10.5281/zenodo.24530700](https://doi.org/10.5281/zenodo.24530700). The archive holds the files that were
recovered (37 of 40, 254 of 288 and 46 of 56 jobs). Download and verify it with

```bash
julia figures/figure_rates/download_data.jl            # all runs, written to data/raw/
julia figures/figure_rates/download_data.jl 2023-02-17 # one run
```

The locations below describe where the files were found when this document was written.

## Where things are

| what | where |
|---|---|
| run submission, run scripts | cluster home `~/pws-cluster-scripts/` (`submit_jobs.jl`, `simple_chemotaxis.jl`) |
| parameters of every submitted job | `~/pws-cluster-scripts/_research/tmp/*.json` (4212 files, Sep 2022 – Mar 2023) |
| raw output, one HDF5 file per job | SMB share `\\sun.amolf.nl\tenwolde`, rclone remote `sun:tenwolde/home-folder/reinhardt/pws-cluster-scripts/data/simple_chemotaxis/<run_name>/` |
| list of the files on the share (3384 files) | `~/compress_files/hdf5_list.txt` |
| aggregation of the HDF5 files | `~/compress_files/analysis.ipynb` (version of 7 Jan 2023) |
| kernel, covariance, input-signal and Gaussian data | `~/pws-cluster-scripts/_research/new_chemotaxis.ipynb` |

The raw HDF5 files were re-compressed in place on the share in September 2025 (`compress_hdf5.py`).

## Pipeline for the PWS data

1. **Submit.** `submit_jobs.jl` expands a dictionary of settings (`run_name`, `cluster_size` = Lmax,
   `n_clusters`, `gradient_steepness`, `smc_samples`, `num_samples`, `duration`, `jobnum`, …) into one JSON file
   per job in `_research/tmp/` and submits each as a 36-CPU job running `simple_chemotaxis.jl`. For example the
   file of one job of run `2023-01-15`:

   ```json
   {"params":{"n_clusters":100,"jobnum":2,"run_name":"2023-01-15","duration":20,"num_samples":1440,
              "cluster_size":5,"methylation":true,"gradient_steepness":0.0003,
              "script":"simple_chemotaxis.jl","algorithm":"smc","smc_samples":256}}
   ```

2. **Run.** `simple_chemotaxis.jl` builds the system with `simple_chemotaxis_system` and estimates the mutual
   information of `num_samples` independent trajectories with `run_parallel(system, SMCEstimate(M), num_samples)`.
   The parameters are fixed in the script: `Kₐ = 2900`, `Kᵢ = 18`, `δf = −2`, `k_R = 0.1`, `k_B = 0.2`,
   `k_Z = 10`, 10 000 CheY, 4 methylation sites per receptor, `m₀ = Lmax/2`, mean ligand 100 µM,
   `τ_L = 1/0.862 s`, velocity noise `√(2·0.862·157.1)` (Mattingly et al.), `k_A = k_Z φ_y / ((1−φ_y) a₀ n_clusters)` with
   `φ_y = 1/6`, output step `dt = 0.1`, SDE step `0.01`. These are the defaults of
   `PWS.chemotaxis_system` in the current package. `n_clusters` is either given directly or computed as
   `R_tot / Lmax` (run `2023-03-27`).

3. **Output.** One HDF5 file per job, named
   `Alg=smc_Clusters=<n>_Duration=20_Job=<j>_M=<M>_Methylation=true_N=<Lmax>_TauL=1.16_g=<g>.hdf5`, with the
   mutual information of every sample in `Samples/MutualInformation` (time points × samples),
   `DiscreteTimes` and `Parameters`. The files were copied to the share listed above.

4. **Aggregate.** `analysis.ipynb` reads the files of a run and, per file, computes the mean
   over the samples and the variance of that mean, `var(mi, dims=2) ./ size(mi, 2)`. Files with the same
   `(g, NClusters)` are then combined: the means are averaged, the sample counts `N` are added and
   **the variances are averaged**. The columns `Mean`, `Var`, `M`, `N` of `2023-02-17_mi_trajectories.csv`
   are the result.

   Consequently `Var` is the variance of the mean of *one file* (1440 trajectories), not of the mean of all
   `N` trajectories. The variance of the full mean is `Var / (N / 1440)`, i.e. `Var / 10` for
   `2023-02-17`. Check: at `T = 20 s`, `g = 0.1 mm⁻¹` the published `Var` is `8.9·10⁻⁴ nats²`, which
   corresponds to a spread of 1.13 nats per trajectory if it is the variance of a 1440-trajectory mean (`Var · 1440`),
   in agreement with the 1.11 nats found when the data is recomputed, and to 3.6 nats if it were the variance of the
   14 400-trajectory mean.

5. **Information rate.** The rate is the mean of the increments `(MI(t) − MI(t−dt)) / dt` of the aggregated
   mutual information (converted to bits) over the stationary window `5 s ≤ t ≤ 20 s`, with
   standard error `std(increments) / √(number of increments)` (`info_rate`, `std` in the CSV files). The version of
   the notebook that was found predates the `Lmax` column of `2023-01-15_pws_info_rates.csv`; the rate
   definition is taken from it *(inferred to be the same)*.

## Runs

All runs use the generic `SMCEstimate` (except `2023-01-06`, which uses PERM), `methylation = true`, the four gradient steepnesses
`g = 0.1, 0.2, 0.3, 0.4 mm⁻¹` (`gradient_steepness = 1e-4 … 4e-4`) and `simple_chemotaxis.jl`.
"Jobs" is the number of JSON files; every job computes "samples" trajectories.

| run | M | clusters | Lmax | duration (s) | samples per job | jobs |
|---|---|---|---|---|---|---|
| `2022-09-26` | 128, 256 | 800 | 6 | 20, 100 | 360 | 108 |
| `2022-09-26_long` | 128 | 800 | 6 | 100 | 360 | 72 |
| `2022-09-27`, `2022-09-27_2` | 128 | 800 | 6 | 20 | 720 | 72, 36 |
| `2022-09-28` | 128 | 800 | 6 | 20 | 720 | 108 |
| `2022-09-28_2`, `2022-09-30` | 512 | 800 | 6 | 20 | 720 | 120, 120 |
| `2022-09-30_3` | 128 | 800 | 6 | 25 | 720 | 40 |
| `2022-10-03` | 128 | 800 | 6 | 25 | 720 | 80 |
| `2022-10-03_2` | 128 | 800 | 6, 15 | 25 | 720 | 80 |
| `2022-10-03_3` | 128 | 10 | 15 | 25 | 720 | 80 |
| `2022-10-04` | 256, 512 | 10 | 15 | 25 | 360 | 168 |
| `2022-10-05` | 1024 | 10 | 15 | 25 | 360 | 296 |
| `2022-10-13` | 2048 | 10 | 15 | 10 | 360 | 80 |
| `2022-12-21` | 256 | 8, 25, 50, 200, 800 | 15 | 20 | 720 | 200 |
| `2023-01-02` | 256 | 20, 30, 40, 50, 60, 80, 100, 200, 300, 400 | 15 | 20 | 720, 1440 | 1248 |
| `2023-01-05` | 512 | 20, 25 | 15 | 20 | 1440 | 16 |
| `2023-01-06` | PERM (10) | 20 | 15 | 20 | 720 | 48 |
| `2023-01-10` | 512 | 25 | 3, 5, 7, 9, 11, 13, 15 | 20 | 1440 | 56 |
| `2023-01-11` | 256 | 400 | 3, 5, 7, 9, 11, 13, 15 | 20 | 1440 | 56 |
| `2023-01-15` | 256 | 30, 40, 50, 60, 100, 200 | 2, 3, 4, 5, 6, 7, 9, 11, 13, 15 | 20 | 1440 | 288 |
| `2023-02-17` | 256 | 400 | 6 | 20 | 1440 | 40 |
| `2023-03-27` | 256 | `150 / Lmax` | 1, 2, 3, 5 | 20 | 720, 1440, 2880 | 800 |

The manifest on the share lists additional files from runs `2022-09-15` to `2022-09-22` for which no job
files were found; it lists 37 files for `2023-02-17` although 40 jobs were submitted (the published
`N = 14400` corresponds to 10 files per gradient steepness).

## Which data file comes from which run

| data file | produced from |
|---|---|
| `2023-02-17_mi_trajectories.csv` | run `2023-02-17`. The columns `NClusters = 400`, `Lmax = 6`, `M = 256`, `N = 14400` match the run exactly (10 jobs × 1440 samples per gradient). |
| `2023-02-17_pws_info_rates_lit.csv` | the same run (400 clusters, Lmax 6). |
| `2022-10-03_2_pws_info_rates_lit.csv` | run `2022-10-03_2` (800 clusters, Lmax 6, `M = 128`) *(inferred from the name and parameters)*. |
| `2023-01-15_pws_info_rates.csv` | runs `2023-01-15` (30–200 clusters) and `2023-01-11` (400 clusters, Lmax 3–15): the combinations of cluster number and Lmax in the file are exactly the union of these two runs *(inferred)*. |
| `2023-01-15_pws_extrapolation.csv` | the extrapolation (`α`, `β`, `extrapolation` per `g` and `Lmax`) of the rates above to small cluster numbers. **The code for this step was not found.** |
| `input_traj_summary.csv` | `new_chemotaxis.ipynb`, cell "input trajectories": the 5/25/75/95 % quantiles and 20 sample trajectories of the ligand concentration. The copy `_research/data/input_traj_summary.csv` (21 Feb 2023) has the same size as the committed file. |
| `cov_data_lit.csv`, `cov_data_fit.csv` | `new_chemotaxis.ipynb`: noise autocovariance of the kinase activity from 10 000 simulated trajectories of 1000 s (`dtimes = 0:0.01:1000`) **at constant ligand concentration** (100 µM), for the literature parameters and for the fitted parameters (`make_params(9, 15)`, i.e. 9 clusters with 15 methylation levels). Recomputing the data with a fluctuating input signal as well gives a 14–23 % larger autocovariance for the literature parameters and does not reproduce the committed file; at constant ligand it does (see [`../figure_kernels/README.md`](../figure_kernels/README.md)). |
| `kern_data_lit.csv`, `kern_data_fit.csv` | `new_chemotaxis.ipynb`: response kernel from the step response (ligand 100 → 110 at `t = 50`) of 100 000 simulated trajectories (`dtimes = 0:0.01:250`). |
| `gaussian_rates.csv` | `new_chemotaxis.ipynb`: the coefficient `β` of the Gaussian information rate `R = β g²` for Mattingly, literature and fit, computed from the Fourier-transformed kernels and noise (`info_rate_coef`). `_research/data/gaussian_rate_beta.csv` (17 Feb 2023) has the same size as the committed file. |

The kernel, covariance and Gaussian data come from a custom simulation of the chemotaxis model inside the
notebook (helper functions `get_system`, `simulate_trajectory`, `make_params`, `merge_cov`,
`merge_kernel`), not from PWS.

## Not found

* The extrapolation code (see above), and which cluster numbers and `Lmax` entered the fit.
* The version of the aggregation notebook that produced the `Lmax` column of `2023-01-15_pws_info_rates.csv`. In the thesis, the fit coefficients of the extrapolation (`R(N_c) = a N_c − b N_c²`) are given in a table (`tbl-fit_coefficients`) that was not found.
* The data processing of the gene expression figure.
* Whether the share contains all files of the runs above (see the note on `2023-02-17`).

## Thesis, notebook and data: where they differ

The method is described in the thesis (chapter 4, "Estimating the kernels from simulations",
<https://manuel-rhdt.github.io/webthesis/chapters/04-chemotaxis.html>). Comparing it with the notebook and the committed data:

| | thesis | notebook / committed data |
|---|---|---|
| noise: ligand | constant, 100 µM | constant (a stochastic input does not reproduce the data) |
| noise: trajectory length | "1 × 10⁴ s" | 1000 s. The relative spread of the single-trajectory autocovariance estimates in `cov_data_*.csv` is about 15 % at short lags, as expected for 1000 s and a correlation time of about 12 s (a 10⁴ s trajectory would give about 5 %) *(estimated)* |
| noise: trajectories | 10⁵ | 10 000 (`Nobs` in `cov_data_*.csv`) |
| kernel: trajectories | 10⁵ | 100 000 (`Nobs` in `kern_data_*.csv`) |
| kernel: protocol | adapt for 50 s at 100 µM, step to 110 µM, record 200 s at 0.01 s | the same (trajectories of 250 s, step at 50 s) |
| kernel: estimator | printed as `K(t) = ln(c_s/c_0) ⟨a(t−t₀) − a(t₀)⟩` | `−⟨a(t) − a(0)⟩ / ln(110/100)`. The committed values (0.66 for the literature and 1.5 for the fitted parameters at `t = 0.5 s`) follow the notebook; the printed formula, which multiplies by `ln(c_s/c_0)`, looks like a typo |
| Fourier transform window | Hanning | the committed β values in `gaussian_rates.csv` are reproduced exactly with a **Lanczos** window and differ by 0.15–0.18 % with a Hanning window; `figure_kernels.jl` plots with Hanning |
| parameters | table of the thesis | the same as the defaults of `PWS.chemotaxis_system` (`φ_Y = 1/6`; the table prints 0.17, the text 0.16) |

In the notebook cell that computes `kern_data_lit.csv`, the system is built from the literature parameters but the
kernel is extracted with `get_kernel(full_traj, params_fitted)`. This is harmless: `get_kernel` only uses
`k_Z / (k_A n_clusters)`, and `k_A n_clusters` is the same for both parameter sets (6.0) because `k_A` is chosen
proportional to `1/n_clusters`.
