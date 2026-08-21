import os
import pandas as pd
import numpy as np
import os 

def merge(table1, table2, table3):
    t1 = pd.read_csv(table1, header=None, dtype=np.float64)
    t2 = pd.read_csv(table2, header=None, dtype=np.float64)
    for i in range(8):
        t1.at[8, i] = t1[i].to_numpy().mean()
        t2.at[8, i] = t2[i].to_numpy().mean()
    t1.to_csv(table1, index=False, header=False)
    t2.to_csv(table2, index=False, header=False)
    res = pd.DataFrame([['NaN' for i in range(8)] for j in range(9)], dtype='str')
    max_value = t1.max(axis=1)
    for i in range(9):
        for j in range(8):
            if t1.at[i, j] == max_value.at[i]:
                res.at[i, j] = f"\\textbf{{{round(t1.at[i, j]*100, 1)}}}$_{{{round(t2.at[i, j]*100, 1)}}}$"
            else:
                res.at[i, j] = f"{round(t1.at[i, j]*100, 1)}$_{{{round(t2.at[i, j]*100, 1)}}}$"
    res.to_csv(table3, index=False, header=False)
    

merge('gap-Y.csv', 'gap_std-Y.csv', 'gap-all-Y.csv')
merge('gap-N.csv', 'gap_std-N.csv', 'gap-all-N.csv')
merge('perf-Y.csv', 'perf_std-Y.csv', 'perf-all-Y.csv')
merge('perf-N.csv', 'perf_std-N.csv', 'perf-all-N.csv')