import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


base_dir = '/work/pi_rsitaram_umass_edu/sohaib/pipelines/traffic_analysis'
data_dir = 'profiling/accuracy'
figures_dir = 'figures'

# models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
models = ['eb6_checkpoint_150epochs_yolov5x',
          'eb6_checkpoint_150epochs_yolov5l',
          'eb6_checkpoint_150epochs_yolov5m',
          'eb6_checkpoint_150epochs_yolov5n',
          'eb5_checkpoint_150epochs_yolov5x',
          'eb5_checkpoint_150epochs_yolov5l',
          'eb5_checkpoint_150epochs_yolov5m',
          'eb5_checkpoint_150epochs_yolov5n',
          'eb4_checkpoint_150epochs_yolov5x',
          'eb4_checkpoint_150epochs_yolov5l',
          'eb4_checkpoint_150epochs_yolov5m',
          'eb4_checkpoint_150epochs_yolov5n',
          'eb3_checkpoint_150epochs_yolov5x',
          'eb3_checkpoint_150epochs_yolov5l',
          'eb3_checkpoint_150epochs_yolov5m',
          'eb3_checkpoint_150epochs_yolov5n',
          'eb2_checkpoint_150epochs_yolov5x',
          'eb2_checkpoint_150epochs_yolov5l',
          'eb2_checkpoint_150epochs_yolov5m',
          'eb2_checkpoint_150epochs_yolov5n',
          'eb1_checkpoint_150epochs_yolov5x',
          'eb1_checkpoint_150epochs_yolov5l',
          'eb1_checkpoint_150epochs_yolov5m',
          'eb1_checkpoint_150epochs_yolov5n',
          'eb0_checkpoint_150epochs_yolov5x',
          'eb0_checkpoint_150epochs_yolov5l',
          'eb0_checkpoint_150epochs_yolov5m',
          'eb0_checkpoint_150epochs_yolov5n',]

# dict of dicts
# key: model name
# value: dict of metrics
metrics = {}

ymax = 0
for model in models:
    cmatrix_file = os.path.join(base_dir, data_dir, f'{model}_confusion_matrix.csv')
    df = pd.read_csv(cmatrix_file)

    ymax = max(ymax, max(df['tp'].to_numpy() + df['fp'].to_numpy()))


for model in models:
    cmatrix_file = os.path.join(base_dir, data_dir, f'{model}_confusion_matrix.csv')
    metrics[model] = {}
    df = pd.read_csv(cmatrix_file)

    plt.plot(df['timestamp'], df['tp'])
    plt.xlabel('Second')
    plt.ylabel('True Positives')
    plt.ylim(0, ymax)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_tp.pdf')
    plt.savefig(filename)
    print(f'\nSaved plot at: {filename}')
    plt.close()

    plt.plot(df['timestamp'], df['fp'])
    plt.xlabel('Second')
    plt.ylabel('False Positives')
    plt.ylim(0, ymax)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_fp.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()

    plt.plot(df['timestamp'], df['fn'])
    plt.xlabel('Second')
    plt.ylabel('False Negatives')
    plt.ylim(0, ymax)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_fn.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()

    plt.plot(df['timestamp'], df['fp'] / (df['fp'] + df['tp']))
    plt.xlabel('Second')
    plt.ylabel('False Positive Rate (FP / (FP+TP))')
    plt.ylim(0, 1.1)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_fpr.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()

    precision = df['tp'] / (df['fp'] + df['tp'])
    plt.plot(df['timestamp'], precision)
    # precision = precision[~np.isnan(precision)]
    # average_precision = np.sum(precision.to_numpy()) / len(precision)
    average_precision = np.sum(df['tp']) / (np.sum(df['fp']) + np.sum(df['tp']))
    print(f'average_precision: {average_precision}')
    # print(f'sum(precision): {np.sum(precision.to_numpy())}')
    # print(f'len(precision): {len(precision)}')
    plt.plot(df['timestamp'], [average_precision] * len(df['timestamp']), color='r')
    plt.xlabel('Second')
    plt.ylabel('Precision (TP / (FP+TP))')
    plt.ylim(0, 1.1)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_precision.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()
    metrics[model]['precision'] = average_precision

    recall = df['tp'] / (df['fn'] + df['tp'])
    plt.plot(df['timestamp'], recall)
    # average_recall = np.sum(recall.to_numpy()) / len(recall)
    average_recall = np.sum(df['tp']) / (np.sum(df['fn']) + np.sum(df['tp']))
    print(f'average_recall: {average_recall}')
    plt.plot(df['timestamp'], [average_recall] * len(df['timestamp']), color='r')
    plt.xlabel('Second')
    plt.ylabel('Recall (TP / (TP+FN))')
    plt.ylim(0, 1.1)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_recall.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()
    metrics[model]['recall'] = average_recall
        
    f1 = precision * recall / ((precision + recall) / 2)
    avg_f1_1 = np.sum(f1.to_numpy()) / len(f1)
    avg_f1_2 = average_precision * average_recall / ((average_precision + average_recall) / 2)
    print(f'avg_f1_1: {avg_f1_1}')
    print(f'avg_f1_2: {avg_f1_2}')
    plt.plot(df['timestamp'], recall)
    plt.plot(df['timestamp'], [avg_f1_1] * len(df['timestamp']), color='r')
    plt.plot(df['timestamp'], [avg_f1_2] * len(df['timestamp']), color='g')
    plt.xlabel('Second')
    plt.ylabel('F1 Score (P*R / ((P+R)/2))')
    plt.ylim(0, 1.1)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_f1.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()

    tpr = df['tp'] / (df['tp'] + df['fp'] + df['fn'])
    # average_tpr = np.sum(tpr) / len(tpr)
    average_tpr = np.sum(df['tp']) / (np.sum(df['tp']) + np.sum(df['fp']) + np.sum(df['fn']))
    print(f'average tpr: {average_tpr}')
    plt.plot(df['timestamp'], tpr)
    plt.plot(df['timestamp'], [average_tpr] * len(df['timestamp']), color='r')
    plt.xlabel('Second')
    plt.ylabel('TPR (TP / (TP+FP+FN))')
    plt.ylim(0, 1.1)
    plt.title(f'Model: {model}')
    filename = os.path.join(figures_dir, f'{model}_tpr.pdf')
    plt.savefig(filename)
    print(f'Saved plot at: {filename}')
    plt.close()
    metrics[model]['average_tpr'] = average_tpr

    if 'tp_tp' in df.columns:
        # Taking tp_tp_ratio per request
        tp_tp_ratio = df['tp_tp'] / (df['tp'])
        # Summing up tp_tp's across trace and averaging instead of averaging per-request
        # tp_tp_ratio
        average_tp_tp_ratio = np.sum(df['tp_tp']) / np.sum(df['tp'])
        print(f'average_tp_tp_ratio: {average_tp_tp_ratio}')
        plt.plot(df['timestamp'], tp_tp_ratio)
        plt.plot(df['timestamp'], [average_tp_tp_ratio] * len(df['timestamp']), color='r')
        plt.xlabel('Second')
        plt.ylabel('TP_TP Ratio (TP_TP / TP)')
        plt.ylim(0, 1.1)
        plt.title(f'Model: {model}')
        filename = os.path.join(figures_dir, f'{model}_tp_tp_ratio.pdf')
        plt.savefig(filename)
        print(f'Saved plot at: {filename}')
        plt.close()
        metrics[model]['tp_tp_ratio'] = average_tp_tp_ratio

    if 'tp_fp' in df.columns:
        tp_fp_ratio = df['tp_fp'] / (df['tp'])
        # average_tp_fp_ratio = np.sum(tp_fp_ratio) / len(tp_fp_ratio)
        average_tp_fp_ratio = np.sum(df['tp_fp']) / np.sum(df['tp'])
        print(f'average_tp_fp_ratio: {average_tp_fp_ratio}')
        plt.plot(df['timestamp'], tp_fp_ratio)
        plt.plot(df['timestamp'], [average_tp_fp_ratio] * len(df['timestamp']), color='r')
        plt.xlabel('Second')
        plt.ylabel('TP_FP Ratio (TP_FP / TP)')
        plt.ylim(0, 1.1)
        plt.title(f'Model: {model}')
        filename = os.path.join(figures_dir, f'{model}_tp_fp_ratio.pdf')
        plt.savefig(filename)
        print(f'Saved plot at: {filename}')
        plt.close()
        metrics[model]['tp_fp_ratio'] = average_tp_fp_ratio

    if 'tp_tp' in df.columns:
        e2e_acc = df['tp_tp'] / (df['tp'] + df['fp'] + df['fn'])
        average_e2e_acc = np.sum(df['tp_tp']) / (np.sum(df['tp']) + np.sum(df['fp']) + np.sum(df['fn']))
        print(f'average_e2e_acc: {average_e2e_acc}')
        plt.plot(df['timestamp'], e2e_acc)
        plt.plot(df['timestamp'], [average_e2e_acc] * len(df['timestamp']), color='r')
        plt.xlabel('Second')
        plt.ylabel('E2E Accuracy (TP_TP / (TP+FP+FN))')
        plt.ylim(0, 1.1)
        plt.title(f'Model: {model}')
        filename = os.path.join(figures_dir, f'{model}_e2e_acc.pdf')
        plt.savefig(filename)
        print(f'Saved plot at: {filename}')
        plt.close()
        metrics[model]['e2e_acc'] = average_e2e_acc


metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.T
metrics_df.to_csv(os.path.join('profiling/profiled', 'accuracy.csv'))
