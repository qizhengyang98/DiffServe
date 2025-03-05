import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


END_TIME = 108000

inPath = 'branching'
outPath = 'branching_smoothed'

fileList = glob.glob(os.path.join(inPath, '*.csv'))

print(fileList)

for inFilename in fileList:
    df = pd.read_csv(inFilename)

    window_size = 5000

    smoothed_car_series = df['car_classification'].rolling(window=window_size).mean()
    smoothed_car_array = smoothed_car_series.dropna().values
    smoothed_car_array = np.ceil(smoothed_car_array).astype(int)
    
    smoothed_face_series = df['facial_recognition'].rolling(window=window_size).mean()
    smoothed_face_array = smoothed_face_series.dropna().values
    smoothed_face_array = np.ceil(smoothed_face_array).astype(int)

    print(f'smoothed_car length: {len(smoothed_car_array)}')
    print(f'smoothed_face length: {len(smoothed_face_array)}')
    
    outDf = pd.DataFrame({'frame': df['frame'].values[:len(smoothed_car_array)],
                          'car_classification': smoothed_car_array,
                          'facial_recognition': smoothed_face_array})
    
    outFilename = os.path.join(outPath, inFilename.split('/')[-1])
    outDf.to_csv(outFilename)
    print(f'DataFrame saved to {outFilename}')
