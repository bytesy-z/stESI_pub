import csv
import sys
from statistics import mean, stdev

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = '/home/zik/UniStuff/FYP/stESI_pub/results/mes_debugico3_/eval/trainings/evaluation_train_simu_SEREEGA_eval_simu_SEREEGA_method_cnn_1d_srcspace_ico3_datasetmes_debug_n_train_80test.csv'

data = {'nmse': [], 'loc error': [], 'auc': [], 'time error': [], 'psnr': []}

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        data['nmse'].append(float(row[1]))
        data['loc error'].append(float(row[2]))
        data['auc'].append(float(row[3]))
        data['time error'].append(float(row[4]))
        data['psnr'].append(float(row[5]))

# compute stats
means = {col: mean(data[col]) for col in data}
stds = {col: stdev(data[col]) for col in data}
mins = {col: min(data[col]) for col in data}
maxs = {col: max(data[col]) for col in data}

metrics = ['NMSE', 'Loc Error', 'AUC', 'Time Error', 'PSNR']
columns = ['nmse', 'loc error', 'auc', 'time error', 'psnr']

print(r'\begin{tabular}{|l|c|c|c|c|}')
print(r'\hline')
print(r'Metric & Mean & Std & Min & Max \\')
print(r'\hline')
for met, col in zip(metrics, columns):
    print(f'{met} & {means[col]:.4f} & {stds[col]:.4f} & {mins[col]:.4f} & {maxs[col]:.4f} \\\\')
print(r'\hline')
print(r'\end{tabular}')