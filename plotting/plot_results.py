import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--cascade', help="The cascade pipeline to plot: ['sdturbo', 'sdxs', 'sdxlltn']")
args = parser.parse_args()

log_folder = '../logs'
end_point = 355
chunk_size = 5
time_frame = np.arange(0, 355, chunk_size)

t_values = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295,0.3,0.305,0.31,0.315,0.32,0.325,0.33,0.335,0.34,0.345,0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.395,0.4,0.405,0.41,0.415,0.42,0.425,0.43,0.435,0.44,0.445,0.45,0.455,0.46,0.465,0.47,0.475,0.48,0.485,0.49,0.495,0.5,0.505,0.51,0.515,0.52,0.525,0.53,0.535,0.54,0.545,0.55,0.555,0.56,0.565,0.57,0.575,0.58,0.585,0.59,0.595,0.6,0.605,0.61,0.615,0.62,0.625,0.63,0.635,0.64,0.645,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.795,0.8,0.805,0.81,0.815,0.82,0.825,0.83,0.835,0.84,0.845,0.85,0.855,0.86,0.865,0.87,0.875,0.88,0.885,0.89,0.895,0.9,0.905,0.91,0.915,0.92,0.925,0.93,0.935,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.99,0.995,1.0]
f_t_values = [0.0,0.037,0.068,0.093,0.115,0.134,0.152,0.169,0.189,0.202,0.215,0.226,0.239,0.25,0.261,0.272,0.282,0.291,0.3,0.309,0.321,0.329,0.337,0.346,0.355,0.365,0.374,0.386,0.396,0.405,0.413,0.422,0.428,0.437,0.444,0.452,0.461,0.469,0.476,0.487,0.497,0.505,0.516,0.525,0.535,0.547,0.559,0.572,0.584,0.601,0.616,0.631,0.644,0.656,0.666,0.672,0.68,0.691,0.699,0.706,0.715,0.724,0.732,0.74,0.747,0.753,0.76,0.764,0.771,0.779,0.784,0.79,0.794,0.801,0.81,0.815,0.821,0.826,0.834,0.84,0.846,0.85,0.856,0.862,0.868,0.875,0.883,0.889,0.896,0.902,0.908,0.914,0.922,0.93,0.937,0.946,0.956,0.967,0.977,0.987,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
map_t_to_ft = dict(zip(t_values, f_t_values))

def scale_to_nearest_005(value):
    return round(np.round(value * 20) / 20, 3)

def closest_index(_list, value):
    return min(range(len(_list)), key=lambda i: abs(_list[i] - value))

def interpolated_fid_score(ratio_of_lightweight, cascade):
    ''' This function returns the FID score corresponding to a given value of
        the ratio of queries served by the lightweight model to the total
        queries served successfully.
    '''
    ratio_of_light = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if cascade == 'sdturbo':
        fid_qaware = [18.55, 17.85, 17.33, 16.98, 16.79, 16.95, 17.26, 17.95, 18.86, 20.17, 22.77]
    elif cascade == 'sdxs':
        fid_qaware = [18.55, 17.68, 17.08, 16.97, 17.13, 17.58, 18.21, 19.23, 20.54, 22.34, 24.86]
    elif cascade == 'sdxlltn':
        fid_qaware = [23.31, 23.14, 23.27, 23.67, 24.42, 25.33, 26.34, 27.53, 29.27, 31.01, 33.27]
    else:
        print(f"Cascade {cascade} is not supported.")
    
    x = ratio_of_light
    y = fid_qaware
    
    # Fit a quadratic curve (degree 2)
    coefficients = np.polyfit(x, y, deg=2)
    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)
    # Generate fitted values
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = polynomial(x_fit)
    index = closest_index(x_fit, ratio_of_lightweight)
    y_interpolated_value = y_fit[index]
    return y_interpolated_value


print(f"Lightweight model in Cascade: {args.cascade}")
"""
Confidence threshold
"""
thres_file = f'{log_folder}/thres_per_second.csv'  
thres_data = pd.read_csv(thres_file)[:end_point]
thres_per_second = thres_data['threshold'].groupby(thres_data['threshold'].index // chunk_size).mean().to_list()
thres_per_second = [map_t_to_ft[scale_to_nearest_005(t)] for t in thres_per_second[:]]
smoothed_thres_per_second = [thres_per_second[0]]
for i in range(1, len(thres_per_second)):
    smoothed_thres_per_second.append(0.3 * thres_per_second[i] + 0.7 * smoothed_thres_per_second[i-1])

smoothed_thres_per_second = (smoothed_thres_per_second-np.min(smoothed_thres_per_second)) / \
                            (np.max(smoothed_thres_per_second) - np.min(smoothed_thres_per_second))

plt.plot(time_frame, smoothed_thres_per_second, label=args.cascade)
plt.yticks(np.arange(0,1,0.1))
plt.xlabel("Time (s)")
plt.ylabel("Confidence Threshold")
plt.grid(True)
plt.legend()
plt.title(f'Confidence threshold of {args.cascade}')
plt.savefig(f"{args.cascade}_confidence_threshold.png")
plt.close()

"""
SLO violation ratio
"""
slo_file = f'{log_folder}/slo_timeouts_per_second.csv'
slo_data = pd.read_csv(slo_file)[:end_point]
timeout_per_second = slo_data['timeout'].groupby(slo_data['timeout'].index // chunk_size).sum()
drop_per_second = slo_data['drop'].groupby(slo_data['drop'].index // chunk_size).sum()
failed_per_second = timeout_per_second + drop_per_second
total_per_second = slo_data['total'].groupby(slo_data['total'].index // chunk_size).sum()
slo_timeouts_per_second = failed_per_second / total_per_second.replace(0, 1)
smoothed_slo_timeouts_per_second = np.convolve(slo_timeouts_per_second, np.ones(chunk_size)/chunk_size, mode='valid')
mean_slo = np.mean(smoothed_slo_timeouts_per_second)
print(f"Mean SLO violation rate: {mean_slo}")

plt.plot(time_frame[:len(smoothed_slo_timeouts_per_second)], smoothed_slo_timeouts_per_second, label=args.cascade)
plt.yticks(np.arange(0,1,0.1))
plt.xlabel("Time (s)")
plt.ylabel("SLO violation ratio")
plt.grid(True)
plt.legend()
plt.title(f'SLO violation ratio of {args.cascade}')
plt.savefig(f"{args.cascade}_slo_violation_ratio.png")
plt.close()

"""
FID scores
"""
if args.cascade == 'sdturbo':
    fid_min, fid_max, scale_factor = 16, 23, 0.08
elif args.cascade == 'sdxs':
    fid_min, fid_max, scale_factor = 16, 25, 0.04
elif args.cascade == 'sdxlltn':
    fid_min, fid_max, scale_factor = 23, 33, -0.1

query_file = f'{log_folder}/query_num_per_second.csv'
query_data = pd.read_csv(query_file)[:end_point]
light_per_second = query_data['sdturbo'].groupby(query_data['sdturbo'].index // chunk_size).sum().to_list()
heavy_per_second  = query_data['sdv15'].groupby(query_data['sdv15'].index // chunk_size).sum().to_list()

ratio_per_second = []
for i in range(len(light_per_second)):
    if light_per_second[i] > 0:
        ratio = heavy_per_second[i] / light_per_second[i]
        ratio_per_second.append(ratio)
ratio_per_second = ratio_per_second / np.max(ratio_per_second) 
fid_per_second = [interpolated_fid_score(1-ft+scale_factor, args.cascade) for ft in ratio_per_second]

smoothed_fid_per_second = np.convolve(fid_per_second, np.ones(chunk_size)/chunk_size, mode='valid')
mean_fid = np.mean(smoothed_fid_per_second)
print(f"Mean FID: {mean_fid}")

plt.plot(time_frame[:len(smoothed_fid_per_second)], smoothed_fid_per_second, label=args.cascade)
plt.yticks(np.arange(fid_min, fid_max, 1))
plt.xlabel("Time (s)")
plt.ylabel("FID")
plt.grid(True)
plt.legend()
plt.title(f'FID of {args.cascade}')
plt.savefig(f"{args.cascade}_fid.png")
plt.close()