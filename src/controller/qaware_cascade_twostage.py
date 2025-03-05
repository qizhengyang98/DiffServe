import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict


MSecsInSec = 1000

class SchedulingAlgorithm:
    def __init__(self, algorithm='base'):
        self.name = algorithm
        
    def print_algorithm(self):
        print(self.name)
        
    def run(self, state):
        actions = []
        return actions
    
class CascadeIterativeAllocator(SchedulingAlgorithm):
    def __init__(self, profiled_runtimes, profiled_throughputs,  total_servers=8):
        self.iteration = 0
        
        # Num of servers allocated to each stage in the cascade
        self.x1 = None
        self.x2 = None
        
        # Batch size for each stage in the cascade
        self.b1 = None
        self.b2 = None
        
        # Confidence threshold for request forwarding
        self.thr = None
        
        # Caching measured demand for logging 
        self.measured_demand = 0
        
        self.profiled_throughputs = profiled_throughputs # TODO: the structure of profiled_throughput, and change the codes which refer to it
        self.profiled_runtimes = profiled_runtimes
        
        self.total_demand = None
        
        self.total_servers = total_servers
        
        self.f_t_df = pd.DataFrame({'t': [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295,0.3,0.305,0.31,0.315,0.32,0.325,0.33,0.335,0.34,0.345,0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.395,0.4,0.405,0.41,0.415,0.42,0.425,0.43,0.435,0.44,0.445,0.45,0.455,0.46,0.465,0.47,0.475,0.48,0.485,0.49,0.495,0.5,0.505,0.51,0.515,0.52,0.525,0.53,0.535,0.54,0.545,0.55,0.555,0.56,0.565,0.57,0.575,0.58,0.585,0.59,0.595,0.6,0.605,0.61,0.615,0.62,0.625,0.63,0.635,0.64,0.645,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.795,0.8,0.805,0.81,0.815,0.82,0.825,0.83,0.835,0.84,0.845,0.85,0.855,0.86,0.865,0.87,0.875,0.88,0.885,0.89,0.895,0.9,0.905,0.91,0.915,0.92,0.925,0.93,0.935,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.99,0.995,1.0],
                                    'f_t': [0.0,0.143,0.2,0.233,0.26,0.287,0.306,0.321,0.333,0.348,0.36,0.371,0.384,0.392,0.403,0.409,0.416,0.425,0.433,0.441,0.449,0.455,0.462,0.469,0.477,0.483,0.489,0.497,0.503,0.509,0.514,0.519,0.524,0.533,0.539,0.545,0.552,0.561,0.567,0.573,0.582,0.589,0.596,0.602,0.608,0.618,0.63,0.64,0.652,0.665,0.69,0.711,0.722,0.734,0.745,0.755,0.764,0.771,0.779,0.784,0.788,0.793,0.799,0.804,0.81,0.816,0.821,0.825,0.828,0.833,0.837,0.842,0.847,0.85,0.855,0.859,0.862,0.866,0.87,0.874,0.878,0.882,0.888,0.892,0.898,0.901,0.906,0.909,0.913,0.918,0.922,0.926,0.93,0.939,0.944,0.952,0.957,0.964,0.972,0.983,0.993,0.996,0.997,0.998,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]})
        
        self.cached_failed_requests = 0
        
    def initialize(self):
        # return the initial batch sizes, ratio of servers, and new threshold i.e., b1, b2, x1, t_new
        self.b1 = 8
        self.b2 = 8
        
        self.x1 = math.ceil((self.total_servers-1) / 2)
        self.x2 = self.total_servers - self.x1 - 1 # Reserving 1 predictor for sink
        
        self.thr = 0.5
        
        requried_workers, batch_sizes_dict = self.prepare_data_structures(
            self.x1, self.x2, self.b1, self.b2
        )
        
        return requried_workers, batch_sizes_dict, self.thr
    
    def iterate(self, total_demand, queue_length_per_task, latencySLO):
        self.total_demand = total_demand
        if self.total_demand == 0:
            return self.initialize()
        
        self.iteration += 1
        if self.iteration % 2 == 0:
            return self.find_batch_sizes_with_fixed_allocation(queue_length_per_task, latencySLO)
        else:
            return self.find_allocation_with_fixed_batch_sizes()
        
    def find_allocation_with_fixed_batch_sizes(self):
        b1 = self.b1
        b2 = self.b2
        thput1 = self.profiled_throughputs['sdturbo', 'sdturbo', b1]
        thput2 = self.profiled_throughputs['sdv15', 'sdv15', b2]
        total_demand = self.total_demand
        self.measured_demand = total_demand
        
        if total_demand == 0:
            x1 = math.ceil(self.total_servers / 2)
            x2 = self.total_servers - x1 -1
        else:
            # step 1: solving T(m1, b1).x1 >= D
            x1 = math.ceil(total_demand / thput1)
            # step 2: solving x1 + x2 <= S
            x2 = self.total_servers - x1 -1
            
        #step 3: solving T(m2,b2).x2 >= D.f(t) or f(t) <= T(m2,b2).x2 / D
        max_f = thput2 * x2 / total_demand
        self.max_f = max_f
        
        # find the maximum threshold t that has f(t)<=max_f
        filtered_f_t = self.f_t_df[self.f_t_df['f_t'] <= max_f]
        if filtered_f_t.empty:
            t_new = None
        else:
            max_indices_per_col = filtered_f_t.apply(pd.Series.argmax)
            argmax_f_t = max_indices_per_col['f_t']
            t_new = self.f_t_df['t'][argmax_f_t]
            
            print(f'x1: {x1}, x2: {x2}')
            print(f"max_f: {max_f}")
            print(f"filtered_f_t: {filtered_f_t}")
            print(f"argmax_f_t: {argmax_f_t}")
            print(f"t_new: {t_new}")
            
        #check whether there is a feasible solution
        if t_new is not None:
            self.x1 = x1
            self.x2 = x2
            self.thr = t_new
        else:
            raise Exception(f"Could not find a feasible solution even with the smallest threhold")
        
        requried_workers, batch_sizes_dict = self.prepare_data_structures(
            self.x1, self.x2, self.b1, self.b2
        )
        
        return requried_workers, batch_sizes_dict, self.thr
    
    def find_batch_sizes_with_fixed_allocation(self, queue_length_per_task, latencySLO):
        total_demand = self.total_demand
        self.measured_demand = total_demand
        
        # step1: solve for x1.T(m1,b1) >= D
        required_thput1 = total_demand / self.x1
        # find min b1 that has thput >= required_thput1
        filtered_thputs = {k:v for k, v in self.profiled_throughputs.items()
                          if k[0] == 'sdturbo' and k[1] == 'sdturbo'}
        b1_new = np.inf
        for k in filtered_thputs:
            if filtered_thputs[k] >= required_thput1 and k[2] < b1_new:
                b1_new = k[2]
        if b1_new == np.inf: # not solving
            required_workers, batch_sizes_dict = self.prepare_data_structures(
                self.x1, self.x2, self.b1, self.b2
            )
            return required_workers, batch_sizes_dict, self.thr
        
        # step2: solve time(b1_new) + time(b2_new) <= slo
        total_queueing_delay_avg = 0
        queueing_delay_per_executor = defaultdict(int)
        for isi in queue_length_per_task:
            queue_length = np.mean(queue_length_per_task[isi])
            avg_queueing_delay = queue_length / total_demand # need to change this into task-specific ewma
            avg_queueing_delay *= MSecsInSec
            
            queueing_delay_per_executor[isi] = avg_queueing_delay
            total_queueing_delay_avg += avg_queueing_delay
            
        b1_new_proc_time = self.profiled_runtimes[('sdturbo', 'sdturbo', b1_new)]
        b1_queueing_delay = queueing_delay_per_executor['sdturbo']
        time_b1_new = b1_new_proc_time + b1_queueing_delay
        
        # solve for time(b2_new) <= slo - time(b1_new)
        b2_queueing_delay = queueing_delay_per_executor['sdv15']
        b2new_proc_time_max = slo - time_b1_new - b2_queueing_delay
        
        b2_new = 1
        potential_batch_and_runtimes = {k:v for k,v in self.profiled_runtimes.items() if k[0] == 'sdv15'}
        for key in potential_batch_and_runtimes:
            (_,_, potential_b2_new) = key
            potential_runtime = potential_batch_and_runtimes[key]
            if potential_b2_new > b2_new and potential_runtime <= b2new_prof_time_max:
                b2_new = potential_b2_new
        
#         # step2: Determine b2_new using AIMD
#         new_timeouts = slo_timeouts
#         old_timeouts = self.cached_failed_requests
#         if new_timeouts - old_timeouts > 0:
#             # Multiplicative decrease
#             b2_new = math.ceil(self.b2 / 2)
#         else:
#             # Additive increase
#             filtered_thputs = {k:v for k, v in self.profiled_throughputs.items()
#                               if k[0]=='sdv15' and k[1]=='sdv15' and v>0}
#             allowed_batch_sizes = list(map(lambda x: x[2], filtered_thputs.keys()))
            
#             b2_arg = allowed_batch_sizes.index(self.b2)
#             b2_new_arg = min(b2_arg + 1, len(allowed_batch_sizes)-1)
            
#             b2_new = allowed_batch_sizes[b2_new_arg]
            
#         self.cached_failed_requests = new_timeouts
        
        # step3: once we have b1 and b2, solve x2.T(m2,b2) >= D.f(t) or f(t) <= x2.T(b2) / D
        max_f = self.x2 * self.profiled_throughputs['sdv15', 'sdv15', b2_new] / total_demand
        self.max_f = max_f
        
        # find the maximum threshold t that has f(t) <= max_f
        filtered_f_t = self.f_t_df[self.f_t_df['f_t'] <= max_f]
        if filtered_f_t.empty:
            t_new = None
        else:
            max_indices_per_col = filtered_f_t.apply(pd.Series.argmax)
            argmax_f_t = max_indices_per_col['f_t']
            t_new = self.f_t_df['t'][argmax_f_t]

            print(f'b1_new: {b1_new}, b2_new: {b2_new}')
            print(f'max_f: {max_f}')
            print(f'filtered_f_t: {filtered_f_t}')
            print(f'argmax_f_t: {argmax_f_t}')
            print(f't_new: {t_new}')
            
        # check whether there is a feasible solution
        if t_new is not None:
            self.b1 = b1_new
            self.b2 = b2_new
            self.thr = t_new
        else:
            raise Exception(f"Could not find a feasible solution even with the smallest threshold")
        
        required_workers, batch_sizes_dict = self.prepare_data_structures(
            self.x1, self.x2, self.b1, self.b2
        )
        return required_workers, batch_sizes_dict, self.thr
        
    
    def prepare_data_structures(self, x1, x2, b1, b2):
        ''' Prepares the data structures required to realize the allocation
        '''
        # TODO: Do not hard-code model and variant names
        required_predictors = {('sdturbo', 'sdturbo'): x1,
                               ('sdv15', 'sdv15'): x2,
                               ('sink', 'sink'): 1}

        # TODO: Do not hard-code model and variant names
        batch_sizes_dict = {('sdturbo', 'sdturbo', b1): 1,
                            ('sdv15', 'sdv15', b2): 1,
                            ('sink', 'sink', 1): 32}

        return required_predictors, batch_sizes_dict
    
    def get_stats(self):
        ''' Reports the statistics of the current iteration
        '''
        return self.iteration, self.x1, self.x2, self.b1, self.b2, self.thr, self.max_f, self.measured_demand