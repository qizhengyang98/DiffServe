import sys, os
sys.path.append('..')
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import gurobipy as gp
from config import get_cas_exec


MSecsInSec = 1000
USecsInSec = 1000 * 1000


class SchedulingAlgorithm:
    def __init__(self, algorithm='base'):
        self.name = algorithm
        
    def print_algorithm(self):
        print(self.name)
        
    def run(self, state):
        actions = []
        return actions
    
class CascadeILPAllocator(SchedulingAlgorithm):
    def __init__(self, profiled_runtimes,  total_servers=1):
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
        
        self.profiled_throughputs = {} 
        self.execution_latencies = {}
        self.profiled_runtimes = profiled_runtimes
        for key in profiled_runtimes:
            (task, variant, batchSize) = key
            runtime = profiled_runtimes[key]
            throughput = batchSize / (runtime / MSecsInSec)
            self.profiled_throughputs[key] = throughput
            
            if runtime == math.inf:
                runtime = 100000
            self.execution_latencies[key] = runtime / MSecsInSec
        
        self.total_servers = total_servers
        
        self.pipeline = get_cas_exec()
        
        self.t_values = [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.105,0.11,0.115,0.12,0.125,0.13,0.135,0.14,0.145,0.15,0.155,0.16,0.165,0.17,0.175,0.18,0.185,0.19,0.195,0.2,0.205,0.21,0.215,0.22,0.225,0.23,0.235,0.24,0.245,0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295,0.3,0.305,0.31,0.315,0.32,0.325,0.33,0.335,0.34,0.345,0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.395,0.4,0.405,0.41,0.415,0.42,0.425,0.43,0.435,0.44,0.445,0.45,0.455,0.46,0.465,0.47,0.475,0.48,0.485,0.49,0.495,0.5,0.505,0.51,0.515,0.52,0.525,0.53,0.535,0.54,0.545,0.55,0.555,0.56,0.565,0.57,0.575,0.58,0.585,0.59,0.595,0.6,0.605,0.61,0.615,0.62,0.625,0.63,0.635,0.64,0.645,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.795,0.8,0.805,0.81,0.815,0.82,0.825,0.83,0.835,0.84,0.845,0.85,0.855,0.86,0.865,0.87,0.875,0.88,0.885,0.89,0.895,0.9,0.905,0.91,0.915,0.92,0.925,0.93,0.935,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.99,0.995,1.0]
        
        if self.pipeline == 'sdturbo':
            self.f_t_values = [0.0,0.143,0.2,0.233,0.26,0.287,0.306,0.321,0.333,0.348,0.36,0.371,0.384,0.392,0.403,0.409,0.416,0.425,0.433,0.441,0.449,0.455,0.462,0.469,0.477,0.483,0.489,0.497,0.503,0.509,0.514,0.519,0.524,0.533,0.539,0.545,0.552,0.561,0.567,0.573,0.582,0.589,0.596,0.602,0.608,0.618,0.63,0.64,0.652,0.665,0.69,0.711,0.722,0.734,0.745,0.755,0.764,0.771,0.779,0.784,0.788,0.793,0.799,0.804,0.81,0.816,0.821,0.825,0.828,0.833,0.837,0.842,0.847,0.85,0.855,0.859,0.862,0.866,0.87,0.874,0.878,0.882,0.888,0.892,0.898,0.901,0.906,0.909,0.913,0.918,0.922,0.926,0.93,0.939,0.944,0.952,0.957,0.964,0.972,0.983,0.993,0.996,0.997,0.998,0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        elif self.pipeline == 'sdxs':
            self.f_t_values = [0.0,0.037,0.068,0.093,0.115,0.134,0.152,0.169,0.189,0.202,0.215,0.226,0.239,0.25,0.261,0.272,0.282,0.291,0.3,0.309,0.321,0.329,0.337,0.346,0.355,0.365,0.374,0.386,0.396,0.405,0.413,0.422,0.428,0.437,0.444,0.452,0.461,0.469,0.476,0.487,0.497,0.505,0.516,0.525,0.535,0.547,0.559,0.572,0.584,0.601,0.616,0.631,0.644,0.656,0.666,0.672,0.68,0.691,0.699,0.706,0.715,0.724,0.732,0.74,0.747,0.753,0.76,0.764,0.771,0.779,0.784,0.79,0.794,0.801,0.81,0.815,0.821,0.826,0.834,0.84,0.846,0.85,0.856,0.862,0.868,0.875,0.883,0.889,0.896,0.902,0.908,0.914,0.922,0.93,0.937,0.946,0.956,0.967,0.977,0.987,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        elif self.pipeline == 'sdxlltn':
            self.f_t_values = [0.0,0.037,0.068,0.093,0.115,0.134,0.152,0.169,0.189,0.202,0.215,0.226,0.239,0.25,0.261,0.272,0.282,0.291,0.3,0.309,0.321,0.329,0.337,0.346,0.355,0.365,0.374,0.386,0.396,0.405,0.413,0.422,0.428,0.437,0.444,0.452,0.461,0.469,0.476,0.487,0.497,0.505,0.516,0.525,0.535,0.547,0.559,0.572,0.584,0.601,0.616,0.631,0.644,0.656,0.666,0.672,0.68,0.691,0.699,0.706,0.715,0.724,0.732,0.74,0.747,0.753,0.76,0.764,0.771,0.779,0.784,0.79,0.794,0.801,0.81,0.815,0.821,0.826,0.834,0.84,0.846,0.85,0.856,0.862,0.868,0.875,0.883,0.889,0.896,0.902,0.908,0.914,0.922,0.93,0.937,0.946,0.956,0.967,0.977,0.987,0.999,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        
        
    def update_num_servers(self, num_servers):
        self.total_servers = num_servers
        
    def initialize(self):
        # return the initial batch sizes, ratio of servers, and new threshold i.e., b1, b2, x1, t_new
        self.b1 = 8
        self.b2 = 8
        
        self.x1 = math.ceil(self.total_servers / 2)
        self.x2 = self.total_servers - self.x1 - 1 # Reserving 1 predictor for sink
        
        self.thr = 1
        
        self.max_f = -1
        
        requried_workers, batch_sizes_dict = self.prepare_data_structures(
            self.x1, self.x2, self.b1, self.b2
        )
        
        return requried_workers, batch_sizes_dict, self.thr
    
    def iterate(self, sysDemand, latencySLO, demand_per_task, queue_length_per_task, do_static=False):
        '''
        return the updated batch sizes, ratio of servers, and new threshold
        i.e., b1, b2, x1, x2, t_new
        '''
        if do_static:
            return self.static_alloc(sysDemand)
        else:
            return self.solve_ilp(sysDemand, latencySLO, demand_per_task, queue_length_per_task)
    
    
    def static_alloc(self, sysDemand):
        if sysDemand == 0:
            return self.initialize()
        
        if self.pipeline == 'sdturbo':
            self.thr = 0.02
            self.x1 = 4
            self.x2 = 12
            self.b1 = 16
            self.b2 = 4
        elif self.pipeline == 'sdxs':
            self.thr = 0.12
            self.x1 = 2
            self.x2 = 14
            self.b1 = 32
            self.b2 = 4
        elif self.pipeline == 'sdxlltn':
            self.thr = 0.02
            self.x1 = 9
            self.x2 = 7
            self.b1 = 4
            self.b2 = 2

        required_predictors, batch_sizes_dict = self.prepare_data_structures(
            self.x1, self.x2, self.b1, self.b2
        )

        return required_predictors, batch_sizes_dict, self.thr
        
        
    def solve_ilp(self, sysDemand, latencySLO, demand_per_task, queue_length_per_task):
        '''
        solves a joint ILP to find the batch sizes, number of servers for 
        lightweight and heavyweight model (assuming a two-stage cascade),
        and new threshold
        '''
        if sysDemand == 0:
            return self.initialize()
        
        self.thr = None
        
        # creating the model
        m = gp.Model('Query-aware diffusion cascade ILP')
        m.setParam("LogToConsole", 0)
        m.setParam("Threads", 12)
        
        # Parameters
        total_servers = self.total_servers
        total_demand = sysDemand
        # allowed_batch_sizes = list(range(1, 65)) # specified in the simulator
        allowed_batch_sizes = [1, 2, 4, 8, 16, 32]
        # TODO: DO not hard-code app name
        app_name = 'sdturbo_sdv15'
        slo = latencySLO
        # SLO is in USecs, we need to convert to secs
        slo /= USecsInSec
        
        # Optimization variables
        x1 = m.addVar(vtype=gp.GRB.INTEGER, name='x1')
        x2 = m.addVar(vtype=gp.GRB.INTEGER, name='x2')
        b1 = m.addVars(allowed_batch_sizes, vtype=gp.GRB.BINARY, name='b1')
        b2 = m.addVars(allowed_batch_sizes, vtype=gp.GRB.BINARY, name='b2')
        # TODO: thr is not really continuous, it is within our sampled set of values
        thr_ind = m.addVars(range(len(self.f_t_values)),
                           vtype=gp.GRB.BINARY, name='threshold_indicator')
        threshold = m.addVar(vtype=gp.GRB.CONTINUOUS, name='threshold')
        
        # Constraints
        m.addConstr(x1 >= 0)
        m.addConstr(x2 >= 0)
        
        # Indicator variable constraints
        m.addConstr(sum(b1[i] for i in allowed_batch_sizes) <= 1)
        m.addConstr(sum(b2[j] for j in allowed_batch_sizes) <= 1)
        m.addConstr(sum(thr_ind[k] for k in range(len(self.f_t_values))) <= 1)
        
        # x1 + x2 <= S (saving one server for the sink)
        m.addConstr(x1 + x2 <= total_servers - 1)
        
        # x1.p(b1) >= D
        m.addConstr(sum(b1[i] * self.profiled_throughputs['sdturbo', 'sdturbo', i]
                       for i in allowed_batch_sizes) * x1 >= total_demand)
        
        # x2.p(b2) >= D.f(t)
        m.addConstr(sum(b2[j] * self.profiled_throughputs['sdv15', 'sdv15', j]
                       for j in allowed_batch_sizes) * x2 >= 
                   total_demand * sum(thr_ind[k] * self.f_t_values[k] 
                                     for k in range(len(self.f_t_values))))
        
        # Defining threshold in terms of its indicator variable
        m.addConstr(threshold == sum(thr_ind[k] * self.t_values[k]
                                    for k in range(len(self.f_t_values))))
        
        # Latency SLO constraint
        exec_latency_b1 = sum(b1[i] * self.execution_latencies['sdturbo', 'sdturbo', i] for i in allowed_batch_sizes)
        exec_latency_b2 = sum(b2[j] * self.execution_latencies['sdv15', 'sdv15', j] for j in allowed_batch_sizes)
        
        # Estimating queuing delays with Little's law and profiled values
        queuing_delay_per_task = defaultdict(int)
        for isi in queue_length_per_task:
            # Using measured queue length as EWMA
            queue_length = queue_length_per_task[isi]
            # Using measured arrival rate as EWMA
            arrival_rate = demand_per_task[isi]
            # Plugging values into Little's law
            avg_queueing_delay = 0 if arrival_rate == 0 else queue_length / arrival_rate
            queuing_delay_per_task[isi] = avg_queueing_delay
            
        queue_latency_b1 = queuing_delay_per_task['sdturbo']
        queue_latency_b2 = queuing_delay_per_task['sdv15']
        
        m.addConstr(exec_latency_b1 + queue_latency_b1 + 
                   exec_latency_b2 + queue_latency_b2 <= slo)
        
        # Optimization objective
        m.setObjective(threshold, gp.GRB.MAXIMIZE)

        # Solve the optimization
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        ilp_overhead = end_time - start_time
        print(f'Time to solve ILP: {ilp_overhead} seconds')

        print(f'total demand: {total_demand}')
        # print(f'queue_latency_b1: {queue_latency_b1}, queue_latency_b2: {queue_latency_b2}')
        # print(f'exec_latency_b1: {exec_latency_b1}, exec_latency_b2: {exec_latency_b2}')
        for i in allowed_batch_sizes:
            # print(f'p(b1[sdturbo, sdturbo, {i}]): {self.profiled_throughputs["sdturbo", "sdturbo", i]}')
            if b1[i].X > 0:
                self.b1 = i
                print(f'x1: {x1.X}, b1: {i}, b1 variable: {b1[i]}')
            if b2[i].X > 0:
                self.b2 = i
                print(f'x2: {x2.X}, b2: {i}, b2 variable: {b2[i]}')
        print(f'threshold: {threshold.X}')
        # time.sleep(0.1)

        # Creating data structures from the solution
        self.thr = threshold.X
        self.x1 = int(x1.X)
        self.x2 = int(x2.X)

        required_predictors, batch_sizes_dict = self.prepare_data_structures(
            self.x1, self.x2, self.b1, self.b2
        )

        return required_predictors, batch_sizes_dict, self.thr
        
    
    def prepare_data_structures(self, x1, x2, b1, b2):
        ''' Prepares the data structures required to realize the allocation
        '''
        # TODO: Do not hard-code model and variant names
        required_predictors = {('sdturbo', 'sdturbo'): x1,
                               ('sdv15', 'sdv15'): x2,
                               ('sink', 'sink'): 1}

        # TODO: Do not hard-code model and variant names
        batch_sizes_dict = {('sdturbo', 'sdturbo'): b1,
                            ('sdv15', 'sdv15'): b2,
                            ('sink', 'sink'): 1}

        return required_predictors, batch_sizes_dict
    
    def get_stats(self):
        ''' Reports the statistics of the current iteration
        '''
        return self.iteration, self.x1, self.x2, self.b1, self.b2, self.thr, self.max_f, self.measured_demand