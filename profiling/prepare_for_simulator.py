import pandas as pd


input_file = 'runtimes/merged/merged_runtimes.csv'
output_file = 'runtimes/merged/simulator_runtimes.csv'

df = pd.read_csv(input_file)

# We replace column names
# model -> Model
# accelerator -> Accel
# batch_size -> batchsize
# 50th_pct_runtimes -> 50th_pct
# 90th_pct_runtimes -> 90th_pct
df.columns = ['', 'Model', 'Accel', 'batchsize', 'avg_runtime', '50th_pct',
              '90th_pct', '95th_pct']

# Insert sink rows
for i in range(1, 64+1):
    new_row = {'Model': ['sink'], 'Accel': ['1080ti'], 'batchsize': [i],
               'avg_runtime': [0.000001], '50th_pct': [0.000001],
               '90th_pct': [0.000001], '95th_pct': [0.000001]}
    new_df = pd.DataFrame(data=new_row)
    df = pd.concat([df, new_df], ignore_index=True)

df = df.drop(df.columns[0], axis=1)

# Currently, we are assuming a homogeneous cluster so we set the same runtimes
# for all accelerators (onnxruntime_cpu, onnxruntime_gpu_pascal, and 
# onnxruntime_gpu_ampere)
df_copy1 = df.copy()
df_copy2 = df.copy()
df_copy3 = df.copy()

df_copy1['Accel'] = 'onnxruntime_cpu'
df_copy2['Accel'] = 'onnxruntime_gpu_pascal'
df_copy3['Accel'] = 'onnxruntime_gpu_ampere'

df = pd.concat([df_copy1, df_copy2, df_copy3], ignore_index=True)
# df = df.drop('Unnamed: 0', axis=1)

# We replace 'ebX' with 'efficientnet-bX' for all variants
df.replace('eb', 'efficientnet-b', inplace=True, regex=True)

# Convert time from seconds to milliseconds
df['avg_runtime'] = df['avg_runtime'] * 1000
df['50th_pct'] = df['50th_pct'] * 1000
df['90th_pct'] = df['90th_pct'] * 1000
df['95th_pct'] = df['95th_pct'] * 1000

print(df)
df.to_csv(output_file)
