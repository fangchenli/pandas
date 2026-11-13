| workload | strings | arm | median ms | pass drift | vs pandas | vs pandas-deferred |
|---|---|---|---|---|---|---|
| W01_sort_prefix | 8 | pandas-eager | 1988.4 | 0% | n/a | 0.16x |
| W01_sort_prefix | 8 | pandas-deferred | 311.9 | 0% | n/a | 1.00x |
| W01_sort_prefix | 8 | proto-compact | 209.2 | 7% | n/a | 1.49x |
| W01_sort_prefix | 8 | proto-tree | 207.4 | 3% | n/a | 1.50x |
| W03_join | 8 | pandas-eager | 514.0 | 0% | n/a | 0.27x |
| W03_join | 8 | pandas-deferred | 138.7 | 2% | n/a | 1.00x |
| W03_join | 8 | proto-compact | 172.6 | 5% | n/a | 0.80x |
| W03_join | 8 | proto-tree | 176.0 | 7% | n/a | 0.79x |
| W04_reindex | 8 | pandas-eager | 1214.1 | 1% | n/a | 0.99x |
| W04_reindex | 8 | pandas-deferred | 1196.0 | 1% | n/a | 1.00x |
| W04_reindex | 8 | proto-compact | 255.4 | 0% | n/a | 4.68x |
| W04_reindex | 8 | proto-tree | 258.1 | 1% | n/a | 4.63x |
| W10_append | 8 | pandas-eager | 1464.8 | 2% | n/a | 0.20x |
| W10_append | 8 | pandas-deferred | 299.8 | 2% | n/a | 1.00x |
| W10_append | 8 | proto-compact | 343.0 | 3% | n/a | 0.87x |
| W10_append | 8 | proto-tree | 342.4 | 1% | n/a | 0.88x |
| W11_chain_scalars | 8 | pandas-eager | 5623.9 | 1% | n/a | 0.05x |
| W11_chain_scalars | 8 | pandas-deferred | 254.9 | 5% | n/a | 1.00x |
| W11_chain_scalars | 8 | proto-compact | 203.5 | 1% | n/a | 1.25x |
| W11_chain_scalars | 8 | proto-tree | 345.6 | 1% | n/a | 0.74x |
| W14_align | 8 | pandas-eager | 1724.0 | 0% | n/a | 0.30x |
| W14_align | 8 | pandas-deferred | 515.0 | 1% | n/a | 1.00x |
| W14_align | 8 | proto-compact | 538.7 | 2% | n/a | 0.96x |
| W14_align | 8 | proto-tree | 536.5 | 2% | n/a | 0.96x |
| deep | 8 | pandas-eager | 2405.6 | 1% | n/a | 0.10x |
| deep | 8 | pandas-deferred | 249.4 | 6% | n/a | 1.00x |
| deep | 8 | proto-compact | 180.2 | 1% | n/a | 1.38x |
| deep | 8 | proto-tree | 187.6 | 3% | n/a | 1.33x |
| full | 8 | pandas-eager | 799.9 | 2% | n/a | 0.29x |
| full | 8 | pandas-deferred | 235.7 | 0% | n/a | 1.00x |
| full | 8 | proto-compact | 229.1 | 0% | n/a | 1.03x |
| full | 8 | proto-tree | 282.9 | 1% | n/a | 0.83x |
| gather_ints | 8 | pandas-eager | 51.0 | 2% | n/a | 0.99x |
| gather_ints | 8 | pandas-deferred | 50.6 | 0% | n/a | 1.00x |
| gather_ints | 8 | proto-compact | 42.9 | 0% | n/a | 1.18x |
| gather_ints | 8 | proto-tree | 43.3 | 0% | n/a | 1.17x |
| narrow | 8 | pandas-eager | 788.4 | 0% | n/a | 0.10x |
| narrow | 8 | pandas-deferred | 80.5 | 2% | n/a | 1.00x |
| narrow | 8 | proto-compact | 65.4 | 1% | n/a | 1.23x |
| narrow | 8 | proto-tree | 74.1 | 1% | n/a | 1.09x |

Arrow memory pool only; NumPy-backed Int64 columns are excluded.

| workload | strings | arm | source MB | retained MB | retained after consume MB |
|---|---|---|---|---|---|
| W01_sort_prefix | 8 | pandas-eager | 1220.7 | 1224.5 | 1224.5 |
| W01_sort_prefix | 8 | pandas-deferred | 1220.7 | 1220.7 | 1068.1 |
| W01_sort_prefix | 8 | proto-compact | 1342.8 | 1342.8 | 1159.7 |
| W01_sort_prefix | 8 | proto-tree | 1342.8 | 1342.8 | 1159.7 |
| W03_join | 8 | pandas-eager | 1220.7 | 153.1 | 153.1 |
| W03_join | 8 | pandas-deferred | 1220.7 | 152.6 | 153.1 |
| W03_join | 8 | proto-compact | 1342.8 | 183.1 | 168.3 |
| W03_join | 8 | proto-tree | 1342.8 | 183.1 | 168.3 |
| W04_reindex | 8 | pandas-eager | 1220.7 | 637.1 | 637.1 |
| W04_reindex | 8 | pandas-deferred | 1220.7 | 637.1 | 637.1 |
| W04_reindex | 8 | proto-compact | 1342.8 | 1342.8 | 1246.0 |
| W04_reindex | 8 | proto-tree | 1342.8 | 1342.8 | 1246.0 |
| W10_append | 8 | pandas-eager | 1220.7 | 612.3 | 612.3 |
| W10_append | 8 | pandas-deferred | 1220.7 | 1220.7 | 1144.6 |
| W10_append | 8 | proto-compact | 1342.8 | 1342.8 | 1251.5 |
| W10_append | 8 | proto-tree | 1342.8 | 1342.8 | 1251.5 |
| W11_chain_scalars | 8 | pandas-eager | 1220.7 | 803.5 | 803.5 |
| W11_chain_scalars | 8 | pandas-deferred | 1220.7 | 1220.7 | 1220.7 |
| W11_chain_scalars | 8 | proto-compact | 1342.8 | 1300.8 | 1300.8 |
| W11_chain_scalars | 8 | proto-tree | 1342.8 | 1300.8 | 1300.8 |
| W14_align | 8 | pandas-eager | 1220.7 | 1224.4 | 1224.4 |
| W14_align | 8 | pandas-deferred | 1220.7 | 1220.7 | 1220.7 |
| W14_align | 8 | proto-compact | 1342.8 | 1342.8 | 1342.8 |
| W14_align | 8 | proto-tree | 1342.8 | 1342.8 | 1342.8 |
| deep | 8 | pandas-eager | 1220.7 | 714.1 | 714.1 |
| deep | 8 | pandas-deferred | 1220.7 | 1220.7 | 1157.4 |
| deep | 8 | proto-compact | 1342.8 | 1315.2 | 1244.9 |
| deep | 8 | proto-tree | 1342.8 | 1315.2 | 1244.9 |
| full | 8 | pandas-eager | 1220.7 | 153.2 | 153.2 |
| full | 8 | pandas-deferred | 1220.7 | 1220.7 | 153.2 |
| full | 8 | proto-compact | 1342.8 | 1319.9 | 168.5 |
| full | 8 | proto-tree | 1342.8 | 1319.9 | 168.5 |
| gather_ints | 8 | pandas-eager | 1220.7 | 0.0 | 0.0 |
| gather_ints | 8 | pandas-deferred | 1220.7 | 0.0 | 0.0 |
| gather_ints | 8 | proto-compact | 1342.8 | 122.1 | 61.0 |
| gather_ints | 8 | proto-tree | 1342.8 | 122.1 | 61.0 |
| narrow | 8 | pandas-eager | 1220.7 | 153.2 | 153.2 |
| narrow | 8 | pandas-deferred | 1220.7 | 1220.7 | 1087.3 |
| narrow | 8 | proto-compact | 1342.8 | 1319.9 | 1159.7 |
| narrow | 8 | proto-tree | 1342.8 | 1319.9 | 1159.7 |
