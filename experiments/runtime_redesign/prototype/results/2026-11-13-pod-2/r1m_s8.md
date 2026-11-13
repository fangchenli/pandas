| workload | strings | arm | median ms | pass drift | vs pandas | vs pandas-deferred |
|---|---|---|---|---|---|---|
| W01_sort_prefix | 8 | pandas | 280.8 | 1% | 1.00x | 0.21x |
| W01_sort_prefix | 8 | pandas-eager | 279.1 | 0% | 1.01x | 0.21x |
| W01_sort_prefix | 8 | pandas-unshared | 59.4 | 1% | 4.72x | 1.00x |
| W01_sort_prefix | 8 | pandas-deferred | 59.5 | 0% | 4.72x | 1.00x |
| W01_sort_prefix | 8 | proto-eager | 269.9 | 0% | 1.04x | 0.22x |
| W01_sort_prefix | 8 | proto-compact | 44.9 | 1% | 6.25x | 1.32x |
| W01_sort_prefix | 8 | proto-tree | 44.5 | 1% | 6.31x | 1.34x |
| W01_sort_prefix | 8 | proto-eager-numpy | 278.9 | 1% | 1.01x | 0.21x |
| W01_sort_prefix | 8 | proto-compact-numpy | 44.2 | 1% | 6.35x | 1.35x |
| W01_sort_prefix | 8 | proto-tree-numpy | 41.6 | 1% | 6.75x | 1.43x |
| W03_join | 8 | pandas | 51.1 | 1% | 1.00x | 0.45x |
| W03_join | 8 | pandas-eager | 52.3 | 1% | 0.98x | 0.44x |
| W03_join | 8 | pandas-unshared | 23.5 | 7% | 2.17x | 0.99x |
| W03_join | 8 | pandas-deferred | 23.2 | 1% | 2.20x | 1.00x |
| W03_join | 8 | proto-eager | 69.1 | 5% | 0.74x | 0.34x |
| W03_join | 8 | proto-compact | 42.4 | 2% | 1.21x | 0.55x |
| W03_join | 8 | proto-tree | 42.8 | 1% | 1.19x | 0.54x |
| W03_join | 8 | proto-eager-numpy | 70.4 | 1% | 0.73x | 0.33x |
| W03_join | 8 | proto-compact-numpy | 40.8 | 0% | 1.25x | 0.57x |
| W03_join | 8 | proto-tree-numpy | 33.5 | 1% | 1.53x | 0.69x |
| W04_reindex | 8 | pandas | 204.6 | 2% | 1.00x | 0.94x |
| W04_reindex | 8 | pandas-eager | 191.7 | 0% | 1.07x | 1.00x |
| W04_reindex | 8 | pandas-unshared | 191.4 | 0% | 1.07x | 1.00x |
| W04_reindex | 8 | pandas-deferred | 192.1 | 0% | 1.06x | 1.00x |
| W04_reindex | 8 | proto-eager | 210.0 | 0% | 0.97x | 0.91x |
| W04_reindex | 8 | proto-compact | 43.8 | 0% | 4.67x | 4.38x |
| W04_reindex | 8 | proto-tree | 43.9 | 0% | 4.66x | 4.38x |
| W04_reindex | 8 | proto-eager-numpy | 211.0 | 1% | 0.97x | 0.91x |
| W04_reindex | 8 | proto-compact-numpy | 38.8 | 0% | 5.27x | 4.95x |
| W04_reindex | 8 | proto-tree-numpy | 39.8 | 1% | 5.15x | 4.83x |
| W10_append | 8 | pandas | 151.6 | 1% | 1.00x | 0.33x |
| W10_append | 8 | pandas-eager | 149.9 | 3% | 1.01x | 0.34x |
| W10_append | 8 | pandas-unshared | 53.4 | 10% | 2.84x | 0.95x |
| W10_append | 8 | pandas-deferred | 50.8 | 1% | 2.99x | 1.00x |
| W10_append | 8 | proto-eager | 114.9 | 3% | 1.32x | 0.44x |
| W10_append | 8 | proto-compact | 67.1 | 1% | 2.26x | 0.76x |
| W10_append | 8 | proto-tree | 66.3 | 0% | 2.29x | 0.77x |
| W10_append | 8 | proto-eager-numpy | 142.9 | 2% | 1.06x | 0.36x |
| W10_append | 8 | proto-compact-numpy | 67.1 | 1% | 2.26x | 0.76x |
| W10_append | 8 | proto-tree-numpy | 68.6 | 3% | 2.21x | 0.74x |
| W11_chain_scalars | 8 | pandas | 403.3 | 2% | 1.00x | 0.17x |
| W11_chain_scalars | 8 | pandas-eager | 402.3 | 4% | 1.00x | 0.17x |
| W11_chain_scalars | 8 | pandas-unshared | 71.5 | 2% | 5.64x | 0.97x |
| W11_chain_scalars | 8 | pandas-deferred | 69.1 | 4% | 5.84x | 1.00x |
| W11_chain_scalars | 8 | proto-eager | 637.4 | 13% | 0.63x | 0.11x |
| W11_chain_scalars | 8 | proto-compact | 52.0 | 0% | 7.76x | 1.33x |
| W11_chain_scalars | 8 | proto-tree | 80.5 | 0% | 5.01x | 0.86x |
| W11_chain_scalars | 8 | proto-eager-numpy | 603.8 | 8% | 0.67x | 0.11x |
| W11_chain_scalars | 8 | proto-compact-numpy | 45.7 | 2% | 8.82x | 1.51x |
| W11_chain_scalars | 8 | proto-tree-numpy | 68.8 | 18% | 5.86x | 1.00x |
| W14_align | 8 | pandas | 255.2 | 4% | 1.00x | 0.40x |
| W14_align | 8 | pandas-eager | 248.1 | 2% | 1.03x | 0.41x |
| W14_align | 8 | pandas-unshared | 103.9 | 0% | 2.46x | 0.98x |
| W14_align | 8 | pandas-deferred | 101.7 | 4% | 2.51x | 1.00x |
| W14_align | 8 | proto-eager | 256.5 | 1% | 1.00x | 0.40x |
| W14_align | 8 | proto-compact | 101.0 | 1% | 2.53x | 1.01x |
| W14_align | 8 | proto-tree | 105.1 | 0% | 2.43x | 0.97x |
| W14_align | 8 | proto-eager-numpy | 258.4 | 0% | 0.99x | 0.39x |
| W14_align | 8 | proto-compact-numpy | 97.9 | 4% | 2.61x | 1.04x |
| W14_align | 8 | proto-tree-numpy | 97.5 | 19% | 2.62x | 1.04x |
| deep | 8 | pandas | 178.5 | 1% | 1.00x | 0.30x |
| deep | 8 | pandas-eager | 188.7 | 4% | 0.95x | 0.28x |
| deep | 8 | pandas-unshared | 77.9 | 6% | 2.29x | 0.68x |
| deep | 8 | pandas-deferred | 52.8 | 3% | 3.38x | 1.00x |
| deep | 8 | proto-eager | 184.1 | 1% | 0.97x | 0.29x |
| deep | 8 | proto-compact | 40.0 | 2% | 4.46x | 1.32x |
| deep | 8 | proto-tree | 40.4 | 0% | 4.42x | 1.31x |
| deep | 8 | proto-eager-numpy | 188.5 | 2% | 0.95x | 0.28x |
| deep | 8 | proto-compact-numpy | 39.3 | 2% | 4.54x | 1.34x |
| deep | 8 | proto-tree-numpy | 40.2 | 3% | 4.44x | 1.32x |
| full | 8 | pandas | 102.2 | 1% | 1.00x | 0.54x |
| full | 8 | pandas-eager | 102.6 | 0% | 1.00x | 0.53x |
| full | 8 | pandas-unshared | 64.4 | 2% | 1.59x | 0.85x |
| full | 8 | pandas-deferred | 54.9 | 0% | 1.86x | 1.00x |
| full | 8 | proto-eager | 107.5 | 2% | 0.95x | 0.51x |
| full | 8 | proto-compact | 54.8 | 1% | 1.86x | 1.00x |
| full | 8 | proto-tree | 61.6 | 3% | 1.66x | 0.89x |
| full | 8 | proto-eager-numpy | 113.6 | 3% | 0.90x | 0.48x |
| full | 8 | proto-compact-numpy | 56.7 | 2% | 1.80x | 0.97x |
| full | 8 | proto-tree-numpy | 63.1 | 2% | 1.62x | 0.87x |
| gather_ints | 8 | pandas | 10.6 | 2% | 1.00x | 1.00x |
| gather_ints | 8 | pandas-eager | 10.8 | 3% | 0.98x | 0.98x |
| gather_ints | 8 | pandas-unshared | 10.6 | 0% | 1.00x | 1.00x |
| gather_ints | 8 | pandas-deferred | 10.6 | 1% | 1.00x | 1.00x |
| gather_ints | 8 | proto-eager | 8.9 | 1% | 1.18x | 1.19x |
| gather_ints | 8 | proto-compact | 8.7 | 1% | 1.22x | 1.22x |
| gather_ints | 8 | proto-tree | 8.8 | 1% | 1.20x | 1.20x |
| gather_ints | 8 | proto-eager-numpy | 20.3 | 5% | 0.52x | 0.52x |
| gather_ints | 8 | proto-compact-numpy | 20.1 | 0% | 0.53x | 0.53x |
| gather_ints | 8 | proto-tree-numpy | 19.5 | 7% | 0.54x | 0.54x |
| narrow | 8 | pandas | 100.1 | 1% | 1.00x | 0.19x |
| narrow | 8 | pandas-eager | 100.1 | 0% | 1.00x | 0.19x |
| narrow | 8 | pandas-unshared | 27.4 | 1% | 3.65x | 0.69x |
| narrow | 8 | pandas-deferred | 19.0 | 2% | 5.28x | 1.00x |
| narrow | 8 | proto-eager | 105.8 | 2% | 0.95x | 0.18x |
| narrow | 8 | proto-compact | 17.4 | 2% | 5.74x | 1.09x |
| narrow | 8 | proto-tree | 18.4 | 3% | 5.44x | 1.03x |
| narrow | 8 | proto-eager-numpy | 107.2 | 1% | 0.93x | 0.18x |
| narrow | 8 | proto-compact-numpy | 18.0 | 0% | 5.56x | 1.05x |
| narrow | 8 | proto-tree-numpy | 19.1 | 9% | 5.23x | 0.99x |

Arrow memory pool only; NumPy-backed Int64 columns are excluded.

| workload | strings | arm | source MB | retained MB | retained after consume MB |
|---|---|---|---|---|---|
| W01_sort_prefix | 8 | pandas | 61.0 | 306.1 | 306.1 |
| W01_sort_prefix | 8 | pandas-eager | 305.2 | 306.1 | 306.1 |
| W01_sort_prefix | 8 | pandas-unshared | 305.2 | 305.2 | 267.0 |
| W01_sort_prefix | 8 | pandas-deferred | 305.2 | 305.2 | 267.0 |
| W01_sort_prefix | 8 | proto-eager | 335.7 | 336.6 | 336.6 |
| W01_sort_prefix | 8 | proto-compact | 335.7 | 335.7 | 289.9 |
| W01_sort_prefix | 8 | proto-tree | 335.7 | 335.7 | 289.9 |
| W01_sort_prefix | 8 | proto-eager-numpy | 305.2 | 306.1 | 306.1 |
| W01_sort_prefix | 8 | proto-compact-numpy | 305.2 | 305.2 | 267.0 |
| W01_sort_prefix | 8 | proto-tree-numpy | 305.2 | 305.2 | 267.0 |
| W03_join | 8 | pandas | 61.0 | 38.3 | 38.3 |
| W03_join | 8 | pandas-eager | 305.2 | 38.3 | 38.3 |
| W03_join | 8 | pandas-unshared | 305.2 | 38.1 | 38.3 |
| W03_join | 8 | pandas-deferred | 305.2 | 38.1 | 38.3 |
| W03_join | 8 | proto-eager | 335.7 | 42.1 | 42.1 |
| W03_join | 8 | proto-compact | 335.7 | 45.8 | 42.1 |
| W03_join | 8 | proto-tree | 335.7 | 45.8 | 42.1 |
| W03_join | 8 | proto-eager-numpy | 305.2 | 38.3 | 38.3 |
| W03_join | 8 | proto-compact-numpy | 305.2 | 38.1 | 38.3 |
| W03_join | 8 | proto-tree-numpy | 305.2 | 38.1 | 38.3 |
| W04_reindex | 8 | pandas | 61.0 | 159.3 | 159.3 |
| W04_reindex | 8 | pandas-eager | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | pandas-unshared | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | pandas-deferred | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | proto-eager | 335.7 | 177.9 | 177.9 |
| W04_reindex | 8 | proto-compact | 335.7 | 335.7 | 311.5 |
| W04_reindex | 8 | proto-tree | 335.7 | 335.7 | 311.5 |
| W04_reindex | 8 | proto-eager-numpy | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | proto-compact-numpy | 305.2 | 305.2 | 286.9 |
| W04_reindex | 8 | proto-tree-numpy | 305.2 | 305.2 | 286.9 |
| W10_append | 8 | pandas | 61.0 | 153.1 | 153.1 |
| W10_append | 8 | pandas-eager | 305.2 | 153.1 | 153.1 |
| W10_append | 8 | pandas-unshared | 305.2 | 305.2 | 286.2 |
| W10_append | 8 | pandas-deferred | 305.2 | 305.2 | 286.2 |
| W10_append | 8 | proto-eager | 335.7 | 168.3 | 168.3 |
| W10_append | 8 | proto-compact | 335.7 | 335.7 | 312.9 |
| W10_append | 8 | proto-tree | 335.7 | 335.7 | 312.9 |
| W10_append | 8 | proto-eager-numpy | 305.2 | 153.1 | 153.1 |
| W10_append | 8 | proto-compact-numpy | 305.2 | 305.2 | 286.2 |
| W10_append | 8 | proto-tree-numpy | 305.2 | 305.2 | 286.2 |
| W11_chain_scalars | 8 | pandas | 61.0 | 201.0 | 201.0 |
| W11_chain_scalars | 8 | pandas-eager | 305.2 | 201.0 | 201.0 |
| W11_chain_scalars | 8 | pandas-unshared | 305.2 | 305.2 | 305.2 |
| W11_chain_scalars | 8 | pandas-deferred | 305.2 | 305.2 | 305.2 |
| W11_chain_scalars | 8 | proto-eager | 335.7 | 221.1 | 221.1 |
| W11_chain_scalars | 8 | proto-compact | 335.7 | 325.2 | 325.2 |
| W11_chain_scalars | 8 | proto-tree | 335.7 | 325.2 | 325.2 |
| W11_chain_scalars | 8 | proto-eager-numpy | 305.2 | 201.0 | 201.0 |
| W11_chain_scalars | 8 | proto-compact-numpy | 305.2 | 305.2 | 305.2 |
| W11_chain_scalars | 8 | proto-tree-numpy | 305.2 | 305.2 | 305.2 |
| W14_align | 8 | pandas | 61.0 | 305.9 | 305.9 |
| W14_align | 8 | pandas-eager | 305.2 | 305.9 | 305.9 |
| W14_align | 8 | pandas-unshared | 305.2 | 305.2 | 305.2 |
| W14_align | 8 | pandas-deferred | 305.2 | 305.2 | 305.2 |
| W14_align | 8 | proto-eager | 335.7 | 336.4 | 336.4 |
| W14_align | 8 | proto-compact | 335.7 | 335.7 | 335.7 |
| W14_align | 8 | proto-tree | 335.7 | 335.7 | 335.7 |
| W14_align | 8 | proto-eager-numpy | 305.2 | 305.9 | 305.9 |
| W14_align | 8 | proto-compact-numpy | 305.2 | 305.2 | 305.2 |
| W14_align | 8 | proto-tree-numpy | 305.2 | 305.2 | 305.2 |
| deep | 8 | pandas | 61.0 | 178.7 | 178.7 |
| deep | 8 | pandas-eager | 305.2 | 178.7 | 178.7 |
| deep | 8 | pandas-unshared | 305.2 | 305.2 | 289.4 |
| deep | 8 | pandas-deferred | 305.2 | 305.2 | 289.4 |
| deep | 8 | proto-eager | 335.7 | 196.5 | 196.5 |
| deep | 8 | proto-compact | 335.7 | 328.8 | 311.3 |
| deep | 8 | proto-tree | 335.7 | 328.8 | 311.3 |
| deep | 8 | proto-eager-numpy | 305.2 | 178.7 | 178.7 |
| deep | 8 | proto-compact-numpy | 305.2 | 305.2 | 289.4 |
| deep | 8 | proto-tree-numpy | 305.2 | 305.2 | 289.4 |
| full | 8 | pandas | 61.0 | 38.3 | 38.3 |
| full | 8 | pandas-eager | 305.2 | 38.3 | 38.3 |
| full | 8 | pandas-unshared | 305.2 | 305.2 | 38.3 |
| full | 8 | pandas-deferred | 305.2 | 305.2 | 38.3 |
| full | 8 | proto-eager | 335.7 | 42.1 | 42.1 |
| full | 8 | proto-compact | 335.7 | 330.0 | 42.1 |
| full | 8 | proto-tree | 335.7 | 330.0 | 42.1 |
| full | 8 | proto-eager-numpy | 305.2 | 38.3 | 38.3 |
| full | 8 | proto-compact-numpy | 305.2 | 305.2 | 38.3 |
| full | 8 | proto-tree-numpy | 305.2 | 305.2 | 38.3 |
| gather_ints | 8 | pandas | 61.0 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-eager | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-unshared | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-deferred | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | proto-eager | 335.7 | 15.3 | 15.3 |
| gather_ints | 8 | proto-compact | 335.7 | 30.5 | 15.3 |
| gather_ints | 8 | proto-tree | 335.7 | 30.5 | 15.3 |
| gather_ints | 8 | proto-eager-numpy | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | proto-compact-numpy | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | proto-tree-numpy | 305.2 | 0.0 | 0.0 |
| narrow | 8 | pandas | 61.0 | 38.3 | 38.3 |
| narrow | 8 | pandas-eager | 305.2 | 38.3 | 38.3 |
| narrow | 8 | pandas-unshared | 305.2 | 305.2 | 271.8 |
| narrow | 8 | pandas-deferred | 305.2 | 305.2 | 271.8 |
| narrow | 8 | proto-eager | 335.7 | 42.1 | 42.1 |
| narrow | 8 | proto-compact | 335.7 | 330.0 | 289.9 |
| narrow | 8 | proto-tree | 335.7 | 330.0 | 289.9 |
| narrow | 8 | proto-eager-numpy | 305.2 | 38.3 | 38.3 |
| narrow | 8 | proto-compact-numpy | 305.2 | 305.2 | 271.8 |
| narrow | 8 | proto-tree-numpy | 305.2 | 305.2 | 271.8 |
