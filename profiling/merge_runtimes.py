import os
import sys
import glob
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import query_yes_no


input_dir = 'runtimes'
output_dir = 'runtimes/merged'

if not(os.path.exists(input_dir)):
    print(f'Input directory does not exist: {input_dir}\nExiting..')
    exit(1)

if not(os.path.exists(output_dir)):
    create_dir = query_yes_no(f'Output directory does not exist: {output_dir}'
                                f'\nCreate directory and proceed?')
    if create_dir:
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')
    else:
        print(f'Directory not created, exiting..')
        exit(1)

input_files = glob.glob(os.path.join(input_dir, '*.csv'))

df_list = []

for input_file in input_files:
    df = pd.read_csv(input_file)
    df_list.append(df)

output_df = pd.concat(df_list, ignore_index=True)
output_df = output_df.sort_values(by=['model', 'batch_size'])
output_df = output_df.drop('Unnamed: 0', axis=1)
output_df.reset_index(drop=True, inplace=True)

# Sanitize values
output_df = output_df.replace('.onnx', '', regex=True)
# output_df = output_df.replace('eb', 'efficientnet-b', regex=True)
output_df = output_df.replace('_checkpoint_150epochs', '', regex=True)
output_df = output_df.replace('_best', '', regex=True)

print(output_df)
output_df.to_csv(os.path.join(output_dir, 'merged_runtimes.csv'))
