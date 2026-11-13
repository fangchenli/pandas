| workload | strings | arm | median ms | pass drift | vs pandas | vs pandas-deferred |
|---|---|---|---|---|---|---|
| W01_sort_prefix | 8 | pandas | 256.0 | 0% | 1.00x | 0.22x |
| W01_sort_prefix | 8 | pandas-eager | 257.1 | 1% | 1.00x | 0.21x |
| W01_sort_prefix | 8 | pandas-unshared | 54.9 | 2% | 4.66x | 1.01x |
| W01_sort_prefix | 8 | pandas-deferred | 55.3 | 1% | 4.63x | 1.00x |
| W01_sort_prefix | 8 | proto-eager | 247.6 | 2% | 1.03x | 0.22x |
| W01_sort_prefix | 8 | proto-compact | 40.6 | 1% | 6.30x | 1.36x |
| W01_sort_prefix | 8 | proto-tree | 40.7 | 0% | 6.29x | 1.36x |
| W01_sort_prefix | 8 | proto-eager-numpy | 253.9 | 0% | 1.01x | 0.22x |
| W01_sort_prefix | 8 | proto-compact-numpy | 40.1 | 0% | 6.38x | 1.38x |
| W01_sort_prefix | 8 | proto-tree-numpy | 40.2 | 1% | 6.37x | 1.38x |
| W01_sort_prefix | 32 | pandas | 903.6 | 12% | 1.00x | 0.06x |
| W01_sort_prefix | 32 | pandas-eager | 869.7 | 9% | 1.04x | 0.07x |
| W01_sort_prefix | 32 | pandas-unshared | 57.8 | 1% | 15.64x | 1.01x |
| W01_sort_prefix | 32 | pandas-deferred | 58.2 | 0% | 15.53x | 1.00x |
| W01_sort_prefix | 32 | proto-eager | 869.1 | 4% | 1.04x | 0.07x |
| W01_sort_prefix | 32 | proto-compact | 40.9 | 4% | 22.07x | 1.42x |
| W01_sort_prefix | 32 | proto-tree | 39.4 | 0% | 22.91x | 1.48x |
| W01_sort_prefix | 32 | proto-eager-numpy | 868.2 | 1% | 1.04x | 0.07x |
| W01_sort_prefix | 32 | proto-compact-numpy | 39.2 | 0% | 23.06x | 1.49x |
| W01_sort_prefix | 32 | proto-tree-numpy | 40.1 | 2% | 22.51x | 1.45x |
| W03_join | 8 | pandas | 48.0 | 1% | 1.00x | 0.45x |
| W03_join | 8 | pandas-eager | 48.3 | 0% | 0.99x | 0.45x |
| W03_join | 8 | pandas-unshared | 22.8 | 5% | 2.10x | 0.95x |
| W03_join | 8 | pandas-deferred | 21.8 | 0% | 2.20x | 1.00x |
| W03_join | 8 | proto-eager | 57.6 | 2% | 0.83x | 0.38x |
| W03_join | 8 | proto-compact | 26.8 | 2% | 1.79x | 0.81x |
| W03_join | 8 | proto-tree | 26.9 | 1% | 1.78x | 0.81x |
| W03_join | 8 | proto-eager-numpy | 56.3 | 1% | 0.85x | 0.39x |
| W03_join | 8 | proto-compact-numpy | 26.6 | 0% | 1.81x | 0.82x |
| W03_join | 8 | proto-tree-numpy | 26.4 | 0% | 1.82x | 0.83x |
| W03_join | 32 | pandas | 115.1 | 19% | 1.00x | 0.20x |
| W03_join | 32 | pandas-eager | 119.0 | 13% | 0.97x | 0.20x |
| W03_join | 32 | pandas-unshared | 23.1 | 1% | 4.98x | 1.02x |
| W03_join | 32 | pandas-deferred | 23.5 | 1% | 4.90x | 1.00x |
| W03_join | 32 | proto-eager | 133.2 | 3% | 0.86x | 0.18x |
| W03_join | 32 | proto-compact | 30.2 | 1% | 3.82x | 0.78x |
| W03_join | 32 | proto-tree | 29.1 | 2% | 3.96x | 0.81x |
| W03_join | 32 | proto-eager-numpy | 126.1 | 3% | 0.91x | 0.19x |
| W03_join | 32 | proto-compact-numpy | 25.1 | 0% | 4.59x | 0.94x |
| W03_join | 32 | proto-tree-numpy | 27.1 | 2% | 4.25x | 0.87x |
| W04_reindex | 8 | pandas | 192.7 | 0% | 1.00x | 0.93x |
| W04_reindex | 8 | pandas-eager | 179.1 | 0% | 1.08x | 1.00x |
| W04_reindex | 8 | pandas-unshared | 178.8 | 0% | 1.08x | 1.01x |
| W04_reindex | 8 | pandas-deferred | 179.9 | 0% | 1.07x | 1.00x |
| W04_reindex | 8 | proto-eager | 195.2 | 2% | 0.99x | 0.92x |
| W04_reindex | 8 | proto-compact | 42.1 | 2% | 4.58x | 4.27x |
| W04_reindex | 8 | proto-tree | 42.6 | 3% | 4.52x | 4.22x |
| W04_reindex | 8 | proto-eager-numpy | 190.3 | 0% | 1.01x | 0.95x |
| W04_reindex | 8 | proto-compact-numpy | 36.4 | 4% | 5.29x | 4.94x |
| W04_reindex | 8 | proto-tree-numpy | 36.0 | 1% | 5.36x | 5.00x |
| W04_reindex | 32 | pandas | 710.7 | 9% | 1.00x | 0.93x |
| W04_reindex | 32 | pandas-eager | 657.7 | 8% | 1.08x | 1.01x |
| W04_reindex | 32 | pandas-unshared | 660.5 | 9% | 1.08x | 1.00x |
| W04_reindex | 32 | pandas-deferred | 661.5 | 1% | 1.07x | 1.00x |
| W04_reindex | 32 | proto-eager | 673.5 | 0% | 1.06x | 0.98x |
| W04_reindex | 32 | proto-compact | 42.6 | 1% | 16.67x | 15.51x |
| W04_reindex | 32 | proto-tree | 42.6 | 1% | 16.67x | 15.52x |
| W04_reindex | 32 | proto-eager-numpy | 658.1 | 0% | 1.08x | 1.01x |
| W04_reindex | 32 | proto-compact-numpy | 36.8 | 4% | 19.31x | 17.97x |
| W04_reindex | 32 | proto-tree-numpy | 35.8 | 2% | 19.86x | 18.49x |
| W10_append | 8 | pandas | 139.1 | 2% | 1.00x | 0.34x |
| W10_append | 8 | pandas-eager | 136.8 | 4% | 1.02x | 0.35x |
| W10_append | 8 | pandas-unshared | 50.4 | 1% | 2.76x | 0.95x |
| W10_append | 8 | pandas-deferred | 47.7 | 1% | 2.92x | 1.00x |
| W10_append | 8 | proto-eager | 108.2 | 2% | 1.29x | 0.44x |
| W10_append | 8 | proto-compact | 62.6 | 2% | 2.22x | 0.76x |
| W10_append | 8 | proto-tree | 63.3 | 5% | 2.20x | 0.75x |
| W10_append | 8 | proto-eager-numpy | 134.2 | 2% | 1.04x | 0.36x |
| W10_append | 8 | proto-compact-numpy | 63.7 | 3% | 2.19x | 0.75x |
| W10_append | 8 | proto-tree-numpy | 64.5 | 4% | 2.16x | 0.74x |
| W10_append | 32 | pandas | 455.0 | 10% | 1.00x | 0.12x |
| W10_append | 32 | pandas-eager | 437.7 | 11% | 1.04x | 0.13x |
| W10_append | 32 | pandas-unshared | 51.2 | 7% | 8.89x | 1.10x |
| W10_append | 32 | pandas-deferred | 56.3 | 3% | 8.09x | 1.00x |
| W10_append | 32 | proto-eager | 395.5 | 1% | 1.15x | 0.14x |
| W10_append | 32 | proto-compact | 64.4 | 3% | 7.06x | 0.87x |
| W10_append | 32 | proto-tree | 66.3 | 1% | 6.86x | 0.85x |
| W10_append | 32 | proto-eager-numpy | 425.0 | 4% | 1.07x | 0.13x |
| W10_append | 32 | proto-compact-numpy | 64.5 | 4% | 7.05x | 0.87x |
| W10_append | 32 | proto-tree-numpy | 65.2 | 1% | 6.98x | 0.86x |
| W11_chain_scalars | 8 | pandas | 348.2 | 1% | 1.00x | 0.15x |
| W11_chain_scalars | 8 | pandas-eager | 349.1 | 2% | 1.00x | 0.15x |
| W11_chain_scalars | 8 | pandas-unshared | 54.3 | 2% | 6.41x | 0.98x |
| W11_chain_scalars | 8 | pandas-deferred | 53.1 | 1% | 6.55x | 1.00x |
| W11_chain_scalars | 8 | proto-eager | 527.5 | 4% | 0.66x | 0.10x |
| W11_chain_scalars | 8 | proto-compact | 123.7 | 1% | 2.81x | 0.43x |
| W11_chain_scalars | 8 | proto-tree | 100.2 | 0% | 3.48x | 0.53x |
| W11_chain_scalars | 8 | proto-eager-numpy | 476.7 | 1% | 0.73x | 0.11x |
| W11_chain_scalars | 8 | proto-compact-numpy | 120.5 | 3% | 2.89x | 0.44x |
| W11_chain_scalars | 8 | proto-tree-numpy | 93.1 | 2% | 3.74x | 0.57x |
| W11_chain_scalars | 32 | pandas | 1517.8 | 14% | 1.00x | 0.08x |
| W11_chain_scalars | 32 | pandas-eager | 1472.7 | 9% | 1.03x | 0.08x |
| W11_chain_scalars | 32 | pandas-unshared | 115.5 | 10% | 13.14x | 1.03x |
| W11_chain_scalars | 32 | pandas-deferred | 119.5 | 3% | 12.70x | 1.00x |
| W11_chain_scalars | 32 | proto-eager | 2056.7 | 0% | 0.74x | 0.06x |
| W11_chain_scalars | 32 | proto-compact | 132.7 | 4% | 11.44x | 0.90x |
| W11_chain_scalars | 32 | proto-tree | 105.0 | 1% | 14.46x | 1.14x |
| W11_chain_scalars | 32 | proto-eager-numpy | 2159.3 | 10% | 0.70x | 0.06x |
| W11_chain_scalars | 32 | proto-compact-numpy | 122.1 | 1% | 12.44x | 0.98x |
| W11_chain_scalars | 32 | proto-tree-numpy | 98.6 | 1% | 15.40x | 1.21x |
| W14_align | 8 | pandas | 196.8 | 3% | 1.00x | 0.28x |
| W14_align | 8 | pandas-eager | 190.7 | 1% | 1.03x | 0.29x |
| W14_align | 8 | pandas-unshared | 55.6 | 1% | 3.54x | 0.98x |
| W14_align | 8 | pandas-deferred | 54.5 | 1% | 3.61x | 1.00x |
| W14_align | 8 | proto-eager | 207.5 | 5% | 0.95x | 0.26x |
| W14_align | 8 | proto-compact | 62.2 | 2% | 3.16x | 0.88x |
| W14_align | 8 | proto-tree | 63.0 | 0% | 3.13x | 0.87x |
| W14_align | 8 | proto-eager-numpy | 202.0 | 4% | 0.97x | 0.27x |
| W14_align | 8 | proto-compact-numpy | 55.4 | 2% | 3.55x | 0.98x |
| W14_align | 8 | proto-tree-numpy | 57.0 | 1% | 3.45x | 0.96x |
| W14_align | 32 | pandas | 647.3 | 30% | 1.00x | 0.09x |
| W14_align | 32 | pandas-eager | 594.2 | 5% | 1.09x | 0.10x |
| W14_align | 32 | pandas-unshared | 56.0 | 1% | 11.56x | 1.04x |
| W14_align | 32 | pandas-deferred | 58.0 | 2% | 11.15x | 1.00x |
| W14_align | 32 | proto-eager | 627.8 | 12% | 1.03x | 0.09x |
| W14_align | 32 | proto-compact | 60.3 | 0% | 10.73x | 0.96x |
| W14_align | 32 | proto-tree | 60.3 | 4% | 10.73x | 0.96x |
| W14_align | 32 | proto-eager-numpy | 635.2 | 5% | 1.02x | 0.09x |
| W14_align | 32 | proto-compact-numpy | 49.6 | 2% | 13.05x | 1.17x |
| W14_align | 32 | proto-tree-numpy | 55.6 | 2% | 11.63x | 1.04x |
| deep | 8 | pandas | 164.8 | 0% | 1.00x | 0.27x |
| deep | 8 | pandas-eager | 168.8 | 4% | 0.98x | 0.26x |
| deep | 8 | pandas-unshared | 62.7 | 1% | 2.63x | 0.70x |
| deep | 8 | pandas-deferred | 44.1 | 1% | 3.73x | 1.00x |
| deep | 8 | proto-eager | 168.8 | 1% | 0.98x | 0.26x |
| deep | 8 | proto-compact | 46.8 | 3% | 3.52x | 0.94x |
| deep | 8 | proto-tree | 47.9 | 2% | 3.44x | 0.92x |
| deep | 8 | proto-eager-numpy | 163.7 | 1% | 1.01x | 0.27x |
| deep | 8 | proto-compact-numpy | 44.5 | 1% | 3.70x | 0.99x |
| deep | 8 | proto-tree-numpy | 47.6 | 0% | 3.46x | 0.93x |
| deep | 32 | pandas | 588.4 | 7% | 1.00x | 0.09x |
| deep | 32 | pandas-eager | 564.1 | 2% | 1.04x | 0.09x |
| deep | 32 | pandas-unshared | 149.8 | 1% | 3.93x | 0.35x |
| deep | 32 | pandas-deferred | 52.2 | 2% | 11.28x | 1.00x |
| deep | 32 | proto-eager | 613.5 | 1% | 0.96x | 0.09x |
| deep | 32 | proto-compact | 46.5 | 0% | 12.65x | 1.12x |
| deep | 32 | proto-tree | 47.3 | 2% | 12.43x | 1.10x |
| deep | 32 | proto-eager-numpy | 615.5 | 21% | 0.96x | 0.08x |
| deep | 32 | proto-compact-numpy | 41.7 | 0% | 14.10x | 1.25x |
| deep | 32 | proto-tree-numpy | 43.2 | 0% | 13.63x | 1.21x |
| full | 8 | pandas | 99.9 | 5% | 1.00x | 0.49x |
| full | 8 | pandas-eager | 99.2 | 1% | 1.01x | 0.50x |
| full | 8 | pandas-unshared | 54.7 | 1% | 1.82x | 0.90x |
| full | 8 | pandas-deferred | 49.3 | 4% | 2.03x | 1.00x |
| full | 8 | proto-eager | 100.0 | 2% | 1.00x | 0.49x |
| full | 8 | proto-compact | 46.0 | 8% | 2.17x | 1.07x |
| full | 8 | proto-tree | 58.5 | 1% | 1.71x | 0.84x |
| full | 8 | proto-eager-numpy | 102.6 | 2% | 0.97x | 0.48x |
| full | 8 | proto-compact-numpy | 47.2 | 0% | 2.11x | 1.04x |
| full | 8 | proto-tree-numpy | 59.4 | 0% | 1.68x | 0.83x |
| full | 32 | pandas | 355.3 | 16% | 1.00x | 0.46x |
| full | 32 | pandas-eager | 358.5 | 0% | 0.99x | 0.46x |
| full | 32 | pandas-unshared | 182.6 | 8% | 1.95x | 0.90x |
| full | 32 | pandas-deferred | 164.9 | 1% | 2.15x | 1.00x |
| full | 32 | proto-eager | 364.6 | 2% | 0.97x | 0.45x |
| full | 32 | proto-compact | 165.1 | 7% | 2.15x | 1.00x |
| full | 32 | proto-tree | 191.9 | 4% | 1.85x | 0.86x |
| full | 32 | proto-eager-numpy | 364.9 | 3% | 0.97x | 0.45x |
| full | 32 | proto-compact-numpy | 161.5 | 1% | 2.20x | 1.02x |
| full | 32 | proto-tree-numpy | 196.4 | 1% | 1.81x | 0.84x |
| gather_ints | 8 | pandas | 7.5 | 10% | 1.00x | 1.06x |
| gather_ints | 8 | pandas-eager | 7.9 | 2% | 0.95x | 1.01x |
| gather_ints | 8 | pandas-unshared | 7.9 | 6% | 0.95x | 1.01x |
| gather_ints | 8 | pandas-deferred | 8.0 | 4% | 0.94x | 1.00x |
| gather_ints | 8 | proto-eager | 7.3 | 10% | 1.03x | 1.09x |
| gather_ints | 8 | proto-compact | 7.1 | 9% | 1.05x | 1.11x |
| gather_ints | 8 | proto-tree | 7.1 | 5% | 1.05x | 1.12x |
| gather_ints | 8 | proto-eager-numpy | 10.0 | 9% | 0.75x | 0.80x |
| gather_ints | 8 | proto-compact-numpy | 13.1 | 3% | 0.57x | 0.61x |
| gather_ints | 8 | proto-tree-numpy | 12.6 | 2% | 0.59x | 0.63x |
| gather_ints | 32 | pandas | 7.1 | 2% | 1.00x | 1.15x |
| gather_ints | 32 | pandas-eager | 7.5 | 5% | 0.95x | 1.09x |
| gather_ints | 32 | pandas-unshared | 7.9 | 7% | 0.90x | 1.03x |
| gather_ints | 32 | pandas-deferred | 8.2 | 11% | 0.87x | 1.00x |
| gather_ints | 32 | proto-eager | 6.8 | 4% | 1.05x | 1.20x |
| gather_ints | 32 | proto-compact | 7.0 | 6% | 1.02x | 1.17x |
| gather_ints | 32 | proto-tree | 6.8 | 7% | 1.05x | 1.20x |
| gather_ints | 32 | proto-eager-numpy | 9.7 | 2% | 0.74x | 0.85x |
| gather_ints | 32 | proto-compact-numpy | 10.5 | 13% | 0.68x | 0.78x |
| gather_ints | 32 | proto-tree-numpy | 9.8 | 4% | 0.73x | 0.84x |
| narrow | 8 | pandas | 95.3 | 2% | 1.00x | 0.17x |
| narrow | 8 | pandas-eager | 97.0 | 2% | 0.98x | 0.17x |
| narrow | 8 | pandas-unshared | 20.7 | 2% | 4.59x | 0.78x |
| narrow | 8 | pandas-deferred | 16.1 | 6% | 5.91x | 1.00x |
| narrow | 8 | proto-eager | 97.8 | 2% | 0.97x | 0.16x |
| narrow | 8 | proto-compact | 13.1 | 2% | 7.27x | 1.23x |
| narrow | 8 | proto-tree | 13.9 | 4% | 6.87x | 1.16x |
| narrow | 8 | proto-eager-numpy | 98.0 | 1% | 0.97x | 0.16x |
| narrow | 8 | proto-compact-numpy | 13.1 | 7% | 7.26x | 1.23x |
| narrow | 8 | proto-tree-numpy | 13.8 | 2% | 6.89x | 1.17x |
| narrow | 32 | pandas | 346.2 | 15% | 1.00x | 0.05x |
| narrow | 32 | pandas-eager | 334.9 | 13% | 1.03x | 0.06x |
| narrow | 32 | pandas-unshared | 40.1 | 1% | 8.62x | 0.47x |
| narrow | 32 | pandas-deferred | 18.8 | 6% | 18.39x | 1.00x |
| narrow | 32 | proto-eager | 350.2 | 3% | 0.99x | 0.05x |
| narrow | 32 | proto-compact | 10.9 | 1% | 31.88x | 1.73x |
| narrow | 32 | proto-tree | 11.8 | 4% | 29.40x | 1.60x |
| narrow | 32 | proto-eager-numpy | 351.4 | 0% | 0.99x | 0.05x |
| narrow | 32 | proto-compact-numpy | 12.8 | 2% | 27.08x | 1.47x |
| narrow | 32 | proto-tree-numpy | 11.6 | 1% | 29.93x | 1.63x |

Arrow memory pool only; NumPy-backed Int64 columns are excluded.

| workload | strings | arm | source MB | retained MB | retained after consume MB |
|---|---|---|---|---|---|
| W01_sort_prefix | 8 | pandas | 61.0 | 306.1 | 306.1 |
| W01_sort_prefix | 8 | pandas-eager | 305.2 | 306.1 | 306.1 |
| W01_sort_prefix | 8 | pandas-unshared | 305.2 | 305.2 | 267.0 |
| W01_sort_prefix | 8 | pandas-deferred | 305.2 | 305.2 | 267.0 |
| W01_sort_prefix | 8 | proto-eager | 544.0 | 336.6 | 336.6 |
| W01_sort_prefix | 8 | proto-compact | 544.0 | 544.0 | 472.0 |
| W01_sort_prefix | 8 | proto-tree | 544.0 | 544.0 | 472.0 |
| W01_sort_prefix | 8 | proto-eager-numpy | 512.0 | 306.1 | 306.1 |
| W01_sort_prefix | 8 | proto-compact-numpy | 512.0 | 512.0 | 448.0 |
| W01_sort_prefix | 8 | proto-tree-numpy | 512.0 | 512.0 | 448.0 |
| W01_sort_prefix | 32 | pandas | 244.1 | 1224.5 | 1224.5 |
| W01_sort_prefix | 32 | pandas-eager | 1220.7 | 1224.5 | 1224.5 |
| W01_sort_prefix | 32 | pandas-unshared | 1220.7 | 1220.7 | 1182.6 |
| W01_sort_prefix | 32 | pandas-deferred | 1220.7 | 1220.7 | 1182.6 |
| W01_sort_prefix | 32 | proto-eager | 2080.0 | 1255.0 | 1255.0 |
| W01_sort_prefix | 32 | proto-compact | 2080.0 | 2080.0 | 2008.0 |
| W01_sort_prefix | 32 | proto-tree | 2080.0 | 2080.0 | 2008.0 |
| W01_sort_prefix | 32 | proto-eager-numpy | 2048.0 | 1224.5 | 1224.5 |
| W01_sort_prefix | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 1984.0 |
| W01_sort_prefix | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 1984.0 |
| W03_join | 8 | pandas | 61.0 | 38.3 | 38.3 |
| W03_join | 8 | pandas-eager | 305.2 | 38.3 | 38.3 |
| W03_join | 8 | pandas-unshared | 305.2 | 38.1 | 38.3 |
| W03_join | 8 | pandas-deferred | 305.2 | 38.1 | 38.3 |
| W03_join | 8 | proto-eager | 544.0 | 42.1 | 42.1 |
| W03_join | 8 | proto-compact | 544.0 | 72.0 | 42.1 |
| W03_join | 8 | proto-tree | 544.0 | 72.0 | 42.1 |
| W03_join | 8 | proto-eager-numpy | 512.0 | 38.3 | 38.3 |
| W03_join | 8 | proto-compact-numpy | 512.0 | 64.0 | 38.3 |
| W03_join | 8 | proto-tree-numpy | 512.0 | 64.0 | 38.3 |
| W03_join | 32 | pandas | 244.1 | 38.3 | 38.3 |
| W03_join | 32 | pandas-eager | 1220.7 | 38.3 | 38.3 |
| W03_join | 32 | pandas-unshared | 1220.7 | 38.1 | 38.3 |
| W03_join | 32 | pandas-deferred | 1220.7 | 38.1 | 38.3 |
| W03_join | 32 | proto-eager | 2080.0 | 42.1 | 42.1 |
| W03_join | 32 | proto-compact | 2080.0 | 72.0 | 42.1 |
| W03_join | 32 | proto-tree | 2080.0 | 72.0 | 42.1 |
| W03_join | 32 | proto-eager-numpy | 2048.0 | 38.3 | 38.3 |
| W03_join | 32 | proto-compact-numpy | 2048.0 | 64.0 | 38.3 |
| W03_join | 32 | proto-tree-numpy | 2048.0 | 64.0 | 38.3 |
| W04_reindex | 8 | pandas | 61.0 | 159.3 | 159.3 |
| W04_reindex | 8 | pandas-eager | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | pandas-unshared | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | pandas-deferred | 305.2 | 159.3 | 159.3 |
| W04_reindex | 8 | proto-eager | 544.0 | 177.9 | 177.9 |
| W04_reindex | 8 | proto-compact | 544.0 | 544.0 | 493.2 |
| W04_reindex | 8 | proto-tree | 544.0 | 544.0 | 493.2 |
| W04_reindex | 8 | proto-eager-numpy | 512.0 | 159.3 | 159.3 |
| W04_reindex | 8 | proto-compact-numpy | 512.0 | 512.0 | 467.9 |
| W04_reindex | 8 | proto-tree-numpy | 512.0 | 512.0 | 467.9 |
| W04_reindex | 32 | pandas | 244.1 | 637.1 | 637.1 |
| W04_reindex | 32 | pandas-eager | 1220.7 | 637.1 | 637.1 |
| W04_reindex | 32 | pandas-unshared | 1220.7 | 637.1 | 637.1 |
| W04_reindex | 32 | pandas-deferred | 1220.7 | 637.1 | 637.1 |
| W04_reindex | 32 | proto-eager | 2080.0 | 655.7 | 655.7 |
| W04_reindex | 32 | proto-compact | 2080.0 | 2080.0 | 2029.2 |
| W04_reindex | 32 | proto-tree | 2080.0 | 2080.0 | 2029.2 |
| W04_reindex | 32 | proto-eager-numpy | 2048.0 | 637.1 | 637.1 |
| W04_reindex | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 2003.9 |
| W04_reindex | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 2003.9 |
| W10_append | 8 | pandas | 61.0 | 153.1 | 153.1 |
| W10_append | 8 | pandas-eager | 305.2 | 153.1 | 153.1 |
| W10_append | 8 | pandas-unshared | 305.2 | 305.2 | 286.2 |
| W10_append | 8 | pandas-deferred | 305.2 | 305.2 | 286.2 |
| W10_append | 8 | proto-eager | 544.0 | 168.3 | 168.3 |
| W10_append | 8 | proto-compact | 544.0 | 544.0 | 494.9 |
| W10_append | 8 | proto-tree | 544.0 | 544.0 | 494.9 |
| W10_append | 8 | proto-eager-numpy | 512.0 | 153.1 | 153.1 |
| W10_append | 8 | proto-compact-numpy | 512.0 | 512.0 | 467.1 |
| W10_append | 8 | proto-tree-numpy | 512.0 | 512.0 | 467.1 |
| W10_append | 32 | pandas | 244.1 | 612.3 | 612.3 |
| W10_append | 32 | pandas-eager | 1220.7 | 612.3 | 612.3 |
| W10_append | 32 | pandas-unshared | 1220.7 | 1220.7 | 1201.7 |
| W10_append | 32 | pandas-deferred | 1220.7 | 1220.7 | 1201.7 |
| W10_append | 32 | proto-eager | 2080.0 | 627.5 | 627.5 |
| W10_append | 32 | proto-compact | 2080.0 | 2080.0 | 2030.9 |
| W10_append | 32 | proto-tree | 2080.0 | 2080.0 | 2030.9 |
| W10_append | 32 | proto-eager-numpy | 2048.0 | 612.3 | 612.3 |
| W10_append | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 2003.1 |
| W10_append | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 2003.1 |
| W11_chain_scalars | 8 | pandas | 61.0 | 201.0 | 201.0 |
| W11_chain_scalars | 8 | pandas-eager | 305.2 | 201.0 | 201.0 |
| W11_chain_scalars | 8 | pandas-unshared | 305.2 | 305.2 | 305.2 |
| W11_chain_scalars | 8 | pandas-deferred | 305.2 | 305.2 | 305.2 |
| W11_chain_scalars | 8 | proto-eager | 544.0 | 221.1 | 221.1 |
| W11_chain_scalars | 8 | proto-compact | 544.0 | 532.0 | 532.0 |
| W11_chain_scalars | 8 | proto-tree | 544.0 | 532.0 | 532.0 |
| W11_chain_scalars | 8 | proto-eager-numpy | 512.0 | 201.0 | 201.0 |
| W11_chain_scalars | 8 | proto-compact-numpy | 512.0 | 512.0 | 512.0 |
| W11_chain_scalars | 8 | proto-tree-numpy | 512.0 | 512.0 | 512.0 |
| W11_chain_scalars | 32 | pandas | 244.1 | 804.1 | 804.1 |
| W11_chain_scalars | 32 | pandas-eager | 1220.7 | 804.1 | 804.1 |
| W11_chain_scalars | 32 | pandas-unshared | 1220.7 | 1220.7 | 1220.7 |
| W11_chain_scalars | 32 | pandas-deferred | 1220.7 | 1220.7 | 1220.7 |
| W11_chain_scalars | 32 | proto-eager | 2080.0 | 824.2 | 824.2 |
| W11_chain_scalars | 32 | proto-compact | 2080.0 | 2068.0 | 2068.0 |
| W11_chain_scalars | 32 | proto-tree | 2080.0 | 2068.0 | 2068.0 |
| W11_chain_scalars | 32 | proto-eager-numpy | 2048.0 | 804.1 | 804.1 |
| W11_chain_scalars | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 2048.0 |
| W11_chain_scalars | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 2048.0 |
| W14_align | 8 | pandas | 61.0 | 305.9 | 305.9 |
| W14_align | 8 | pandas-eager | 305.2 | 305.9 | 305.9 |
| W14_align | 8 | pandas-unshared | 305.2 | 305.2 | 305.2 |
| W14_align | 8 | pandas-deferred | 305.2 | 305.2 | 305.2 |
| W14_align | 8 | proto-eager | 544.0 | 336.4 | 336.4 |
| W14_align | 8 | proto-compact | 544.0 | 544.0 | 544.0 |
| W14_align | 8 | proto-tree | 544.0 | 544.0 | 544.0 |
| W14_align | 8 | proto-eager-numpy | 512.0 | 305.9 | 305.9 |
| W14_align | 8 | proto-compact-numpy | 512.0 | 512.0 | 512.0 |
| W14_align | 8 | proto-tree-numpy | 512.0 | 512.0 | 512.0 |
| W14_align | 32 | pandas | 244.1 | 1223.8 | 1223.8 |
| W14_align | 32 | pandas-eager | 1220.7 | 1223.8 | 1223.8 |
| W14_align | 32 | pandas-unshared | 1220.7 | 1220.7 | 1220.7 |
| W14_align | 32 | pandas-deferred | 1220.7 | 1220.7 | 1220.7 |
| W14_align | 32 | proto-eager | 2080.0 | 1254.3 | 1254.3 |
| W14_align | 32 | proto-compact | 2080.0 | 2080.0 | 2080.0 |
| W14_align | 32 | proto-tree | 2080.0 | 2080.0 | 2080.0 |
| W14_align | 32 | proto-eager-numpy | 2048.0 | 1223.8 | 1223.8 |
| W14_align | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 2048.0 |
| W14_align | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 2048.0 |
| deep | 8 | pandas | 61.0 | 178.7 | 178.7 |
| deep | 8 | pandas-eager | 305.2 | 178.7 | 178.7 |
| deep | 8 | pandas-unshared | 305.2 | 305.2 | 289.4 |
| deep | 8 | pandas-deferred | 305.2 | 305.2 | 289.4 |
| deep | 8 | proto-eager | 544.0 | 196.5 | 196.5 |
| deep | 8 | proto-compact | 544.0 | 535.6 | 492.2 |
| deep | 8 | proto-tree | 544.0 | 535.6 | 492.2 |
| deep | 8 | proto-eager-numpy | 512.0 | 178.7 | 178.7 |
| deep | 8 | proto-compact-numpy | 512.0 | 512.0 | 470.3 |
| deep | 8 | proto-tree-numpy | 512.0 | 512.0 | 470.3 |
| deep | 32 | pandas | 244.1 | 714.8 | 714.8 |
| deep | 32 | pandas-eager | 1220.7 | 714.8 | 714.8 |
| deep | 32 | pandas-unshared | 1220.7 | 1220.7 | 1204.9 |
| deep | 32 | pandas-deferred | 1220.7 | 1220.7 | 1204.9 |
| deep | 32 | proto-eager | 2080.0 | 732.7 | 732.7 |
| deep | 32 | proto-compact | 2080.0 | 2071.6 | 2028.2 |
| deep | 32 | proto-tree | 2080.0 | 2071.6 | 2028.2 |
| deep | 32 | proto-eager-numpy | 2048.0 | 714.8 | 714.8 |
| deep | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 2006.3 |
| deep | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 2006.3 |
| full | 8 | pandas | 61.0 | 38.3 | 38.3 |
| full | 8 | pandas-eager | 305.2 | 38.3 | 38.3 |
| full | 8 | pandas-unshared | 305.2 | 305.2 | 38.3 |
| full | 8 | pandas-deferred | 305.2 | 305.2 | 38.3 |
| full | 8 | proto-eager | 544.0 | 42.1 | 42.1 |
| full | 8 | proto-compact | 544.0 | 537.9 | 42.1 |
| full | 8 | proto-tree | 544.0 | 537.9 | 42.1 |
| full | 8 | proto-eager-numpy | 512.0 | 38.3 | 38.3 |
| full | 8 | proto-compact-numpy | 512.0 | 512.0 | 38.3 |
| full | 8 | proto-tree-numpy | 512.0 | 512.0 | 38.3 |
| full | 32 | pandas | 244.1 | 153.3 | 153.3 |
| full | 32 | pandas-eager | 1220.7 | 153.3 | 153.3 |
| full | 32 | pandas-unshared | 1220.7 | 1220.7 | 153.3 |
| full | 32 | pandas-deferred | 1220.7 | 1220.7 | 153.3 |
| full | 32 | proto-eager | 2080.0 | 157.1 | 157.1 |
| full | 32 | proto-compact | 2080.0 | 2073.9 | 157.1 |
| full | 32 | proto-tree | 2080.0 | 2073.9 | 157.1 |
| full | 32 | proto-eager-numpy | 2048.0 | 153.3 | 153.3 |
| full | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 153.3 |
| full | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 153.3 |
| gather_ints | 8 | pandas | 61.0 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-eager | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-unshared | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-deferred | 305.2 | 0.0 | 0.0 |
| gather_ints | 8 | proto-eager | 544.0 | 15.3 | 15.3 |
| gather_ints | 8 | proto-compact | 544.0 | 32.0 | 15.3 |
| gather_ints | 8 | proto-tree | 544.0 | 32.0 | 15.3 |
| gather_ints | 8 | proto-eager-numpy | 512.0 | 0.0 | 0.0 |
| gather_ints | 8 | proto-compact-numpy | 512.0 | 0.0 | 0.0 |
| gather_ints | 8 | proto-tree-numpy | 512.0 | 0.0 | 0.0 |
| gather_ints | 32 | pandas | 244.1 | 0.0 | 0.0 |
| gather_ints | 32 | pandas-eager | 1220.7 | 0.0 | 0.0 |
| gather_ints | 32 | pandas-unshared | 1220.7 | 0.0 | 0.0 |
| gather_ints | 32 | pandas-deferred | 1220.7 | 0.0 | 0.0 |
| gather_ints | 32 | proto-eager | 2080.0 | 15.3 | 15.3 |
| gather_ints | 32 | proto-compact | 2080.0 | 32.0 | 15.3 |
| gather_ints | 32 | proto-tree | 2080.0 | 32.0 | 15.3 |
| gather_ints | 32 | proto-eager-numpy | 2048.0 | 0.0 | 0.0 |
| gather_ints | 32 | proto-compact-numpy | 2048.0 | 0.0 | 0.0 |
| gather_ints | 32 | proto-tree-numpy | 2048.0 | 0.0 | 0.0 |
| narrow | 8 | pandas | 61.0 | 38.3 | 38.3 |
| narrow | 8 | pandas-eager | 305.2 | 38.3 | 38.3 |
| narrow | 8 | pandas-unshared | 305.2 | 305.2 | 271.8 |
| narrow | 8 | pandas-deferred | 305.2 | 305.2 | 271.8 |
| narrow | 8 | proto-eager | 544.0 | 42.1 | 42.1 |
| narrow | 8 | proto-compact | 544.0 | 537.9 | 471.7 |
| narrow | 8 | proto-tree | 544.0 | 537.9 | 471.7 |
| narrow | 8 | proto-eager-numpy | 512.0 | 38.3 | 38.3 |
| narrow | 8 | proto-compact-numpy | 512.0 | 512.0 | 452.8 |
| narrow | 8 | proto-tree-numpy | 512.0 | 512.0 | 452.8 |
| narrow | 32 | pandas | 244.1 | 153.3 | 153.3 |
| narrow | 32 | pandas-eager | 1220.7 | 153.3 | 153.3 |
| narrow | 32 | pandas-unshared | 1220.7 | 1220.7 | 1187.3 |
| narrow | 32 | pandas-deferred | 1220.7 | 1220.7 | 1187.3 |
| narrow | 32 | proto-eager | 2080.0 | 157.1 | 157.1 |
| narrow | 32 | proto-compact | 2080.0 | 2073.9 | 2007.7 |
| narrow | 32 | proto-tree | 2080.0 | 2073.9 | 2007.7 |
| narrow | 32 | proto-eager-numpy | 2048.0 | 153.3 | 153.3 |
| narrow | 32 | proto-compact-numpy | 2048.0 | 2048.0 | 1988.8 |
| narrow | 32 | proto-tree-numpy | 2048.0 | 2048.0 | 1988.8 |
