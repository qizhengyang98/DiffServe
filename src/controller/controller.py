# This is the Controller microservice
import argparse
import sys, os
sys.path.append('..')
import csv
import grpc
import logging
import pandas as pd
import numpy as np
import pickle
import threading
import time
from concurrent import futures
from enum import Enum
from common.app import App, AppNode, registerApplication
from protos import controller_pb2, controller_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from protos import worker_pb2, worker_pb2_grpc
from qaware_cascade_twostage import CascadeIterativeAllocator
from qaware_cascade_ILP import CascadeILPAllocator
from config import get_cas_exec, set_cas_exec


EWMA_WINDOW = 20
EWMA_ALPHA = 0.7
DEFAULT_EWMA_DECAY = 2.1
USECONDS_IN_SEC = 1000 * 1000
SLO_FACTOR = 1


class AllocationPolicy(Enum):
    ROUND_ROBIN = 1
    TAIL_HEAVY = 2
    ILP_COST = 3
    INFERLINE = 4
    CAS_ITER = 5
    CAS_ALL_HEAVY = 6
    CAS_ALL_LIGHT = 7
    CAS_STATIC = 8


class WorkerEntry:
    def __init__(self, IP: str, port: str, hostID: str, connection: grpc.insecure_channel,
                 model: str=None, task: str=None, appID: str=None):
        self.IP = IP
        self.port = port
        self.hostID = hostID
        self.connection = connection
        self.model = model
        self.task = task
        self.appID = appID
        self.currentLoad = 0
        self.queueSize = 0
        self.infer_level = 0
        self.conf_thres = 0
        self.batch_size = 1
        self.scheduler = 'ddim'
        self.onCUDA = None

    def setModel(self, model, task, appID):
        self.model = model
        self.task = task
        self.appID = appID
        
    def setLevel(self, infer_level):
        self.infer_level = infer_level
        
    def setConfThres(self, conf_thres):
        self.conf_thres = conf_thres
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size


class Controller(controller_pb2_grpc.ControllerServicer):
    def __init__(self, allocationPolicy: AllocationPolicy):
        self.lbIP = None
        self.lbPort = None
        self.lbConnection = None
        # key: hostID, value: WorkerEntry
        self.workers = {}
        # Time in seconds after which controller is invoked
        self.period = 1
        
        # overall system demand and EWMA value for the entire system
        self.total_demand_history = [] # List to store system-wide demand over time
        self.ewma_demand_per_task = {'sdturbo': [0], 'sdv15': [0]}
        self.ewma_queue_length_per_task = {'sdturbo': [0], 'sdv15': [0]}
        self.demand_per_task_history = {'sdturbo': [0], 'sdv15': [0]}
        self.queue_length_per_task_history = {'sdturbo': [0], 'sdv15': [0]}
        self.system_ewma = 0 # system-wide EWMA for total demand
        self.coming_query_per_task = {'sdturbo': 0, 'sdv15': 0, 'sink': 0}

        self.allocationPolicy = allocationPolicy
        self.allocationMetadata = {}
        self.cas_alg = None
        self.conf_thres = 1

        # TODO: update execution and branching profiles based on real-time data
        self.executionProfiles = None
        self.branchingProfiles = None
        
        # save request processing time, for the computation of SLO violations
        self.queriesStartTime = {}
        self.queriesIntermediateTime = {}
        self.queriesEndTime = {}
        self.slo_timeouts = {'succeed': 0, 'timeout': 0, 'drop': 0, 'total': 0}
        self.save_slo_timeouts_per_second = f'../../logs/slo_timeouts_per_second.csv'
        self.save_threshold_per_second = f"../../logs/thres_per_second.csv"
        self.save_query_num_per_second = f"../../logs/query_num_per_second.csv"

        # TODO: allow all models, perhaps use model families as well?
        # self.allocatedModels = {'yolov5m': 0, 'eb6': 0, 'sink': 0}
        self.allocatedModels = {'sdturbo': 0, 'sdv15': 0, 'sink': 0}
        self.lightWeightModels = ['sdxl-turbo', 'sdv15-lcm', 'sdxl-lcm', 
                          'sdxl-lightning', 'sdxs', 'tinysd', 'sdturbo']
        self.heavyWeightModels =  ['sd21', 'sdxl', 'sdv15']
        
        PIPELINE = get_cas_exec()
        logging.info(f"Current cascade pipeline: {PIPELINE}")
        if PIPELINE == 'sdturbo':
            executionProfiles = pd.read_csv('../../profiling/stable_diffusion_runtimes_sdturbo.csv')
            self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdturbo_sdv15.json')]
        elif PIPELINE == 'sdxs':
            executionProfiles = pd.read_csv('../../profiling/stable_diffusion_runtimes_sdxs.csv')
            self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdxs_sdv15.json')]
        elif PIPELINE == 'sdxlltn':
            executionProfiles = pd.read_csv('../../profiling/stable_diffusion_runtimes_sdxlltn.csv')
            self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdxlltn_sdxl.json')]
            
        self.profiled_runtimes = {}
        for bs in [1, 2, 4, 8, 16, 32]:
            for model in ['sdturbo', 'sdv15']:
                runtime = executionProfiles.loc[(executionProfiles['Model'].str.contains(model)) & 
                                                (executionProfiles['batchsize']==bs) & 
                                                (executionProfiles['Accel']=='onnxruntime_cpu')]['avg_runtime'].values[0]
                self.profiled_runtimes[(model, model, bs)] = runtime
               
        try:
            eventLoopThread = threading.Thread(target=self.eventLoop)
            eventLoopThread.daemon = True
            eventLoopThread.start()
            # while eventLoopThread.is_alive():
            #     eventLoopThread.join(1)
        except KeyboardInterrupt:
            sys.exit(1)
        
    
    def WorkerSetup(self, request, context):
        try:
            logging.info(f'Trying to establish GRPC connection with worker..')
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                workerIP = splitContext[1]
            else:
                workerIP = 'localhost'
            connection = grpc.insecure_channel(f'{workerIP}:{request.hostPort}')
            workerEntry = WorkerEntry(IP=workerIP, port=request.hostPort,
                                      hostID=request.hostID, connection=connection,
                                      model=None)
            self.workers[request.hostID] = workerEntry
            logging.info(f'Established GRPC connection with worker (hostID: '
                         f'{request.hostID}, IP: {workerIP}, port: '
                         f'{request.hostPort})')
            
            return controller_pb2.RegisterWorkerResponse(lbIP=self.lbIP,
                                                         lbPort=self.lbPort)
        except Exception as e:
            message = f'Exception while setting up worker: {str(e)}'
            logging.exception(message)
            return controller_pb2.RegisterWorkerResponse(lbIP=None, lbPort=None,
                                                         message=message)
    

    def LBSetup(self, request, context):
        try:
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                lbIP = splitContext[1]
            else:
                lbIP = 'localhost'
            logging.info(f'Trying to establish GRPC connection with load balancer..')
            connection = grpc.insecure_channel(f'{lbIP}:{request.lbPort}')
            self.lbConnection = connection
            self.lbIP = lbIP
            self.lbPort = request.lbPort
            logging.info(f'Established GRPC connection with load balancer '
                         f'(IP: {lbIP}, port: {request.lbPort})')
            
            return controller_pb2.RegisterLBResponse(message='Done!')
        
        except Exception as e:
            message = f'Exception while setting up load balancer: {str(e)}'
            logging.exception(message)
            return controller_pb2.RegisterLBResponse(message=message)
    

    def eventLoop(self):
        clockCounter = 0
        prev_slo_timeouts = {k:v for k,v in self.slo_timeouts.items()}
        check_start_time = time.time()
        while True:
            self.checkLBHeartbeat()

            cur_workers = {k:v for k,v in self.workers.items()}
            for hostID in cur_workers:
                # worker = self.workers[hostID]
                worker = cur_workers[hostID]
                self.checkWorkerHeartbeat(hostID, worker)
                
            # Estimate the future system demand
            self.computeSystemEWMA()
            
            # This doesn't necessary have to run every time Controller checks
            # heartbeats
            if clockCounter % 5 == 0:
                self.allocateResources()
            clockCounter += 1
            
            # compute slo timeout per second
            self.computeSLOTimeoutsPerSec()
            logging.info(f'slo_timeouts: {self.slo_timeouts}')
            logging.info(f'prev_slo_timeouts: {prev_slo_timeouts}')
            logging.info(f'length of queriesStartTime: {len(self.queriesStartTime)}, '
                         f'length of queriesIntermediateTime: {len(self.queriesIntermediateTime)}, '
                         f'length of queriesEndTime: {len(self.queriesEndTime)}')
            # save slo timeouts and threshold
            if self.system_ewma > 0:
                self.saveResultsToCSV(self.slo_timeouts, prev_slo_timeouts)
                self.coming_query_per_task = {'sdturbo': 0, 'sdv15': 0, 'sink': 0}
            prev_slo_timeouts = {k:v for k,v in self.slo_timeouts.items()}

            time_difference = time.time() - check_start_time
            sleep_time = self.period - time_difference
            time.sleep(sleep_time)
            check_start_time = time.time()
            logging.debug(f'Woke up from sleep {sleep_time}, running eventLoop again..')
            
            
    def computeSLOTimeoutsPerSec(self):
        # update dropped queries
        expire_time = self.apps[0].getLatencySLO() / USECONDS_IN_SEC * 2 * SLO_FACTOR
        
        popped = []
        for requestID in self.queriesStartTime:
            time_diff = time.time() - self.queriesStartTime[requestID]
            # logging.info(f"time_diff: {time_diff}, expire_time: {expire_time}")
            if expire_time < time_diff:
                popped.append(requestID)
                self.slo_timeouts['drop'] += 1
                self.slo_timeouts['total'] += 1
        for requestID in popped:
            self.queriesStartTime.pop(requestID)
        
    def saveResultsToCSV(self, slo_timeouts, prev_slo_timeouts):
        slo_timeouts_per_second = []
        for key in slo_timeouts:
            slo_timeouts_per_second.append(slo_timeouts[key] - prev_slo_timeouts[key])
            
        for csv_config in [(self.save_slo_timeouts_per_second, ['succeed', 'timeout', 'drop', 'total'], slo_timeouts_per_second), 
                           (self.save_threshold_per_second, ['threshold'], [self.conf_thres]),
                           (self.save_query_num_per_second, ['sdturbo', 'sdv15'], [self.coming_query_per_task['sdturbo'], self.coming_query_per_task['sdv15']])]:
            csv_name, csv_hearder, csv_new_row = csv_config
            file_exists = os.path.exists(csv_name)
            with open(csv_name, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists or os.stat(csv_name).st_size==0:
                    writer.writerow(csv_hearder)
                writer.writerow(csv_new_row)
            
    def computeSystemEWMA(self):
        total_system_demand = 0
        queue_length_per_task = {'sdturbo': 0, 'sdv15': 0, 'sink': 0}
        demand_per_task = {'sdturbo': 0, 'sdv15': 0, 'sink': 0}
        
        cur_workers = {k:v for k,v in self.workers.items()}
        for hostID in cur_workers:
            # worker = self.workers[hostID]
            worker = cur_workers[hostID]
            # total_system_demand += worker.demand
            if worker.model:
                queue_length_per_task[worker.model] += worker.queueSize
                demand_per_task[worker.model] += worker.currentLoad
                
        demand_per_task['sdturbo'] = (len(self.queriesStartTime) + len(self.queriesEndTime)) # add a factor to increase measured demand to reduce slo violation
        demand_per_task['sdv15'] = len(self.queriesIntermediateTime)
        total_system_demand = demand_per_task['sdturbo'] * 1.2 # add a factor to increase measured demand to reduce slo violation
        if total_system_demand > 0:
            for task in self.queue_length_per_task_history:
                self.queue_length_per_task_history[task].append(queue_length_per_task[task])
                self.demand_per_task_history[task].append(demand_per_task[task])
                
                if len(self.queue_length_per_task_history[task]) > EWMA_WINDOW:
                    self.queue_length_per_task_history[task] = self.queue_length_per_task_history[task][1:]
                if len(self.demand_per_task_history[task]) > EWMA_WINDOW:
                    self.demand_per_task_history[task] = self.demand_per_task_history[task][1:]
                
                # compute ewma
                df = pd.DataFrame({'demand': self.demand_per_task_history[task]})
                ewma = df.ewm(com=DEFAULT_EWMA_DECAY).mean()
                ewma_value = ewma['demand'].to_list()[-1]
                self.ewma_demand_per_task[task] = ewma_value
                
                df = pd.DataFrame({'queue_length': self.queue_length_per_task_history[task]})
                ewma = df.ewm(com=DEFAULT_EWMA_DECAY).mean()
                ewma_value = ewma['queue_length'].to_list()[-1]
                self.ewma_queue_length_per_task[task] = ewma_value
                
            self.total_demand_history.append(total_system_demand)
            if len(self.total_demand_history) > EWMA_WINDOW:
                self.total_demand_history = self.total_demand_history[1:]
            # Apply EWMA to the total system demand
            df = pd.DataFrame({'demand': self.total_demand_history})
            ewma = df.ewm(com=DEFAULT_EWMA_DECAY).mean()
            ewma_value = ewma['demand'].to_list()[-1]
            self.system_ewma = ewma_value
        logging.info(f"System-wide Demand = {total_system_demand}, EWMA = {self.system_ewma}")
        
        
    def getSystemDemand(self):
        # return the current system-wide demand and EWMA
        return self.system_ewma
    
    
    def allocateResources(self):
        ''' Run the resource allocation algorithm with the appropriate policy
        '''
        if self.allocationPolicy == AllocationPolicy.ROUND_ROBIN:
            self.allocateByRoundRobin()
        elif self.allocationPolicy == AllocationPolicy.TAIL_HEAVY:
            self.allocateByTailHeavy()
        elif self.allocationPolicy == AllocationPolicy.ILP_COST:
            self.allocateByCostILP()
        elif self.allocationPolicy == AllocationPolicy.INFERLINE:
            self.allocateByInferLine()
        elif self.allocationPolicy == AllocationPolicy.CAS_ITER:
            self.allocateByCascadeAlg()
        elif self.allocationPolicy == AllocationPolicy.CAS_ALL_HEAVY:
            self.allocateByAllHeavy()
        elif self.allocationPolicy == AllocationPolicy.CAS_ALL_LIGHT:
            self.allocateByAllLight()
        elif self.allocationPolicy == AllocationPolicy.CAS_STATIC:
            self.allocateByCascadeAlg(do_static=True)
        else:
            raise Exception(f'Unknown allocation policy: {self.allocationPolicy}')
        return
    
    
    def allocateByAllHeavy(self):
        logging.info(f'AllocatedModels before ReAlloc: {self.allocatedModels}')
        total_server = len(self.workers) if len(self.workers)>1 else 1
        required_workers = {('sdturbo', 'sdturbo'): 0,
                           ('sdv15', 'sdv15'): total_server-1,
                           ('sink', 'sink'): 1}
        
        cur_workers = {k:v for k,v in self.workers.items()}
        for hostID in cur_workers:
            # worker = self.workers[hostID]
            worker = cur_workers[hostID]
        
            if worker.model is None:
                # logging.info(f'Model loaded at worker {hostID}: {worker.model}')
                if worker.onCUDA:
                    modelToLoad, infer_level = self.getModelByCascadeAlg(required_workers)
                else:
                    modelToLoad = 'sink'
                    infer_level = 2
                
                if modelToLoad is None:
                    # logging.info(f'No need to change the loaded model on all workers')
                    modelToLoad = worker.model
                try:
                    batch_size = 1
                    logging.info(f'Trying to load model on worker {hostID}: [{worker.model} -> {modelToLoad}], batch size: {batch_size}')
                    self.loadModelOnWorker(worker, modelToLoad, infer_level=infer_level, batch_size=batch_size)
                except Exception as e:
                    logging.exception(f'Error while loading model {modelToLoad} on '
                                        f'worker {hostID}: {e}')
                        
        logging.info(f'AllocatedModels after ReAlloc: {self.allocatedModels}')
        return
    
    def allocateByAllLight(self):
        logging.info(f'AllocatedModels before ReAlloc: {self.allocatedModels}')
        total_server = len(self.workers) if len(self.workers)>1 else 1
        required_workers = {('sdturbo', 'sdturbo'): total_server-1,
                           ('sdv15', 'sdv15'): 0,
                           ('sink', 'sink'): 1}
        self.conf_thres = 0
        
        cur_workers = {k:v for k,v in self.workers.items()}
        for hostID in cur_workers:
            # worker = self.workers[hostID]
            worker = cur_workers[hostID]
        
            if worker.model is None:
                # logging.info(f'Model loaded at worker {hostID}: {worker.model}')
                if worker.onCUDA:
                    modelToLoad, infer_level = self.getModelByCascadeAlg(required_workers)
                else:
                    modelToLoad = 'sink'
                    infer_level = 2
                    
                if modelToLoad is None:
                    # logging.info(f'No need to change the loaded model on all workers')
                    modelToLoad = worker.model
                try:
                    batch_size = 1
                    logging.info(f'Trying to load model on worker {hostID}: [{worker.model} -> {modelToLoad}], batch size: {batch_size}')
                    self.loadModelOnWorker(worker, modelToLoad, infer_level=infer_level, batch_size=batch_size)
                except Exception as e:
                    logging.exception(f'Error while loading model {modelToLoad} on '
                                        f'worker {hostID}: {e}')
                        
        logging.info(f'AllocatedModels after ReAlloc: {self.allocatedModels}')
        return
    
    def allocateByCascadeAlg(self, do_static=False):
        app = self.apps[0]
        logging.info(f'AllocatedModels before ReAlloc: {self.allocatedModels}')
        
        if self.cas_alg is None:
            self.cas_alg = CascadeILPAllocator(self.profiled_runtimes)
            required_workers, batch_sizes_dict, self.conf_thres = self.cas_alg.initialize()
            logging.info(f'AllocatedModels Required: {required_workers}')
        else:
            slo_timeouts = 0 # TODO: get the slo_timeouts
            logging.info(f'queue_lengh_per_task_ewma: {self.ewma_queue_length_per_task}, demand_per_task_ewma: {self.ewma_demand_per_task}')
            
            num_workers = len(self.workers) if len(self.workers)>1 else 1
            self.cas_alg.update_num_servers(num_workers)
            required_workers, batch_sizes_dict, self.conf_thres = self.cas_alg.iterate(self.system_ewma, app.getLatencySLO(), self.ewma_demand_per_task, self.ewma_queue_length_per_task, do_static)
            logging.info(f'AllocatedModels Required: {required_workers}')
        
        cur_workers = {k:v for k,v in self.workers.items()}
        for hostID in cur_workers:
            # worker = self.workers[hostID]
            worker = cur_workers[hostID]
        
            # if worker.model is None or doRealloc:
                # logging.info(f'Model loaded at worker {hostID}: {worker.model}')
            if worker.onCUDA:
                modelToLoad, infer_level = self.getModelByCascadeAlg(required_workers)
            else:
                modelToLoad = 'sink'
                infer_level = 2

            if modelToLoad is None:
                # logging.info(f'No need to change the loaded model on all workers')
                modelToLoad = worker.model
                infer_level = worker.infer_level
            try:
                batch_size = batch_sizes_dict[modelToLoad, modelToLoad]
                logging.info(f'Trying to load model on worker {hostID}: [{worker.model} -> {modelToLoad}], batch size: {batch_size}')
                self.loadModelOnWorker(worker, modelToLoad, infer_level=infer_level, batch_size=batch_size)
            except Exception as e:
                logging.exception(f'Error while loading model {modelToLoad} on '
                                        f'worker {hostID}: {e}')
                        
        logging.info(f'AllocatedModels after ReAlloc: {self.allocatedModels}')
        logging.info(f'Confidence threshold: {self.conf_thres}')
        return
        
    def getModelByCascadeAlg(self, required_workers):
        changes = 0
        for model in self.allocatedModels:
            if model == 'sink':
                continue
            if self.allocatedModels[model] < required_workers[model, model]:
                changes += 1
                break
        if changes == 0: # no changes, no need to load models
            return None, None
        if model in self.lightWeightModels:
            infer_level = 0
        elif model in self.heavyWeightModels:
            infer_level = 1
        else:
            infer_level = 2 # for sink
        return model, infer_level
    

    def allocateByRoundRobin(self):
        ''' Perform round-robin resource allocation of models to workers
        '''
        for hostID in self.workers:
            worker = self.workers[hostID]
            
            if worker.model is None:
                logging.info(f'No model loaded at worker {hostID}')
                modelToLoad, infer_level = self.getModelByRoundRobin()
                try:
                    logging.info(f'Trying to load model {modelToLoad} on worker {hostID}')
                    self.loadModelOnWorker(worker, modelToLoad, infer_level=infer_level)
                except Exception as e:
                    logging.exception(f'Error while loading model {modelToLoad} on '
                                        f'worker {hostID}: {e}')
                    
        return
    

    def allocateByTailHeavy(self):
        raise Exception(f'Tail-heavy allocation policy not yet implemented, '
                        f'controller crashing')
        return
    

    def allocateByCostILP(self):
        raise Exception(f'Cost ILP allocation policy not yet implemented, '
                        f'controller crashing')
        return
    

    def allocateByInferLine(self):
        if 'inferline_initiated' not in self.allocationMetadata:
            self.allocateByInferLineInitial()
        else:
            self.allocateByInferLinePeriodic()
        return
    

    def allocateByInferLineInitial(self):
        # TODO: Assuming one app for now
        app = self.apps[0]
        allocationPlan = {}

        tasks = app.getAllTasks()
        for task in tasks:
            modelVariants = app.getModelVariantsFromTaskName(task)
            # TODO: Get the most accurate model variant
            # TODO: Assuming they are sorted by accuracy, but this is a weak
            #       assumption
            logging.warning(f'allocateByInferLineInitial assumes models are '
                            f'sorted by accuracy')
            modelVariant = modelVariants[-1]

            # An allocation plan is a dict with the following definiton:
            # key:    (model variant, batch size, hardware)
            # value:  replicas

            batchSize = 1

            # TODO: remove hard-coded value of hardware, find best hardware instead
            #       hardware = bestHardware(modelVariant)
            hardware = '1080ti'

            allocationPlan[(modelVariant, batchSize, hardware)] = 1

        serviceTime = app.getServiceTimeForAllocation(allocationPlan=allocationPlan,
                                                      executionProfiles=self.executionProfiles)
        logging.info(f'service time: {serviceTime} micro-seconds')

        if serviceTime >= app.getLatencySLO():
            raise Exception(f'allocateByInferLineInitial(): No allocation possible '
                            f'as serviceTime ({serviceTime}) is more than application '
                            f'latency SLO ({app.getLatencySLO()})')
        else:
            totalWorkers = len(self.workers)
            totalWorkers = 20
            assignedWorkers = sum(allocationPlan.values())

            logging.info(f'assignedWorkers: {assignedWorkers}, totalWorkers: {totalWorkers}')

            while assignedWorkers < totalWorkers:
                task = app.findMinThroughput(allocationPlan=allocationPlan,
                                             executionProfiles=self.executionProfiles,
                                             branchingProfiles=self.branchingProfiles)
                modelVariants = app.getModelVariantsFromTaskName(task)
                modelVariant = modelVariants[-1]
                batchSize = 1
                # TODO: change hard-coded hardware, use same hardware as before
                hardware = '1080ti'
                key = (modelVariant, batchSize, hardware)

                if key not in allocationPlan:
                    raise Exception(f'Error! Key {key} not already in allocation plan')
                
                allocationPlan[key] += 1
                assignedWorkers += 1
                logging.info(f'Incremented replica for {key} by 1')
        
        self.allocationMetadata['inferline_initiated'] = True
        return
    

    def allocateByInferLinePeriodic(self):
        raise Exception(f'allocateByInferLinePeriodic not yet implemented, '
                        f'controller crashing')

        
    def checkLBHeartbeat(self):
        try:
            connection = self.lbConnection
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            message = 'Still alive?'
            request = load_balancer_pb2.LBHeartbeat(message=message)
            response = stub.LBAlive(request)
            logging.info(f'Heartbeat from load balancer received')
        except Exception as e:
            logging.warning('No heartbeat from load balancer')

    
    def checkWorkerHeartbeat(self, hostID: str, worker: WorkerEntry):
        try:
            connection = worker.connection
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            message = 'Still alive?'
            request = worker_pb2.HeartbeatRequest(message=message)
            response = stub.Heartbeat(request)
            worker.currentLoad = response.queriesSinceHeartbeat
            worker.queueSize = response.queueSize
            worker.onCUDA = response.onCUDA
            branchingSinceHeartbeat = pickle.loads(response.branchingSinceHeartbeat)
            queriesTimestampSinceHearbeat = pickle.loads(response.queriesTimestampSinceHearbeat)
            self.coming_query_per_task[worker.model] += worker.currentLoad
            logging.info(f'Heartbeat from worker {hostID} received, model variant: '
                         f'{worker.model}, currentLoad: {worker.currentLoad}, total '
                         f'queries received: {response.totalQueries}, queue size: '
                         f'{worker.queueSize}, cuda available: {worker.onCUDA}, '
                         f'branching since heartbeat: {len(branchingSinceHeartbeat)}, '
                         f'queries timestamp since heartbeat: {len(queriesTimestampSinceHearbeat)}')
            # if lightweight model worker (infer_level=0), then save new reuqests processing time
            # if sink worker (infer_level=2), then update the requests processing time
            if worker.infer_level in [0, 1]:
                for requestID in queriesTimestampSinceHearbeat:
                    if requestID in self.queriesEndTime:
                        endTime = self.queriesEndTime.pop(requestID)
                        processingTime = endTime - queriesTimestampSinceHearbeat[requestID]
                        if processingTime > self.apps[0].getLatencySLO() / USECONDS_IN_SEC * SLO_FACTOR:
                            self.slo_timeouts['timeout'] += 1
                        else:
                            self.slo_timeouts['succeed'] += 1
                        self.slo_timeouts['total'] += 1
                    else:
                        if requestID in self.queriesStartTime:
                            self.queriesIntermediateTime[requestID] = queriesTimestampSinceHearbeat[requestID]
                        else:
                            self.queriesStartTime[requestID] = queriesTimestampSinceHearbeat[requestID]
                        
            elif worker.infer_level == 2:
                for requestID in queriesTimestampSinceHearbeat:
                    if requestID in self.queriesStartTime:
                        startTime = self.queriesStartTime.pop(requestID)
                        processingTime = queriesTimestampSinceHearbeat[requestID] - startTime
                        if processingTime > self.apps[0].getLatencySLO() / USECONDS_IN_SEC * SLO_FACTOR:
                            self.slo_timeouts['timeout'] += 1
                        else:
                            self.slo_timeouts['succeed'] += 1
                        self.slo_timeouts['total'] += 1
                        
                        if requestID in self.queriesIntermediateTime:
                            self.queriesIntermediateTime.pop(requestID)
                    else:
                        self.queriesEndTime[requestID] = queriesTimestampSinceHearbeat[requestID]
            
        except Exception as e:
            # TODO: remove worker after certain number of missed heartbeats?
            logging.warning(f'No heartbeat from worker: {hostID}')
            logging.exception(f'Exception while checking heartbeat for worker {hostID}: {e}')


    def loadModelOnWorker(self, worker: WorkerEntry, model: str, infer_level: int, batch_size: int):
        ''' Loads the given model on a worker
        '''
        try:
            previousModel = worker.model
            connection = worker.connection
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            # TODO: hard-coded application index
            app = self.apps[0]
            appID = app.appID
            task = app.findTaskFromModelVariant(model)
            childrenTasks = pickle.dumps(app.getChildrenTasks(task))
            labelToChildrenTasks = pickle.dumps(app.getLabelToChildrenTasksDict(task))

            request = worker_pb2.LoadModelRequest(modelName=model,
                                                  schedulerName=worker.scheduler,
                                                  infer_level=infer_level,
                                                  conf_thres=self.conf_thres,
                                                  batch_size=batch_size,
                                                  applicationID=appID,
                                                  task=task,
                                                  childrenTasks=childrenTasks,
                                                  labelToChildrenTasks=labelToChildrenTasks)
            response = stub.LoadModel(request)
            logging.info(f'LOAD_MODEL_RESPONSE from host {worker.hostID}: {response.response}, '
                         f'{response.message}')
            
            # Model loaded without any errors
            if response.response == 0:
                self.allocatedModels[model] += 1
                if previousModel is not None:
                    self.allocatedModels[previousModel] -= 1
            # If there is an error while loading model, raise Exception
            else:
                raise Exception(f'Error occurred while loading model {model} on worker '
                                f'{worker.hostID}: {response.message}')
            
            worker.setModel(model, task, appID)
            worker.setLevel(infer_level)
            worker.setConfThres(self.conf_thres)
            worker.setBatchSize = batch_size
        except Exception as e:
            raise e


    def getModelByRoundRobin(self):
        ''' Returns a model such that self.allocatedModels would have equal
            number of allocated workers for all tasks in the pipeline
            (except the sink which only needs one worker)
        '''
        minValue = sys.maxsize
        minModel = None
        for model in self.allocatedModels:
            if self.allocatedModels[model] < minValue:
                # We only want one copy of the sink for each application
                if model == 'sink' and self.allocatedModels[model] == 1:
                    continue
                
                minValue = self.allocatedModels[model]
                minModel = model
        if model in self.lightWeightModels:
            infer_level = 0
        elif model in self.heavyWeightModels:
            infer_level = 1
        return minModel, infer_level
    
    
    def adjustConfThres(self):
        '''
        Adjust the threshold given the estimated demands, throughput of loaded models,
        and the total number of workers.
        Determine the number of 1st-workers and 2nd-workers, and the confidence threshold
        '''
        EWMA = self.system_ewma
        app = self.apps[0]
        allocationPlan = {"workers1stLevel":0, "workers2ndLevel":0}
        tasks = app.getAllTasks()
        serviceTime = app.getServiceTimeForAllocation(allocationPlan=allocationPlan,
                                                      executionProfiles=self.executionProfiles)
        logging.info(f'service time: {serviceTime} micro-seconds')
        if serviceTime >= app.getLatencySLO():
            raise Exception(f'allocateByInferLineInitial(): No allocation possible '
                            f'as serviceTime ({serviceTime}) is more than application '
                            f'latency SLO ({app.getLatencySLO()})')
        else:
            totalWorkers = len(self.workers)
            assignedWorkers = sum(allocationPlan.values())

            logging.info(f'assignedWorkers: {assignedWorkers}, totalWorkers: {totalWorkers}')

            allocationPlan["workers1stLevel"] = 1
            allocationPlan["workers2ndLevel"] = totalWorkers - allocationPlan["workers1stLevel"]
            keep_searching = True
            while keep_searching:
                thput1st, thput2nd = app.findAllocThroughput(allocationPlan=allocationPlan)
                batchSize = 4
                
                flag = 0
                num1stWorkers = allocationPlan["workers1stLevel"]
                num2ndWorkers = allocationPlan["workers2ndLevel"]
                conf_thres = num1stWorkers / totalWorkers
                if num1stWorkers * thput1st > EWMA:
                    flag += 0.5
                if num2ndWorkers * thput2nd > EWMA * (1-conf_thres):
                    flag += 0.5
                
                if flag == 1:
                    keep_searching = False
                    continue
                
                allocationPlan["workers1stLevel"] += 1
                allocationPlan["workers2ndLevel"] -= 1
                logging.info(f'Incremented replica for workers1stLevel by 1')
        
        conf_thres = allocationPlan["workers1stLevel"] / len(self.workers)
        return conf_thres
    

def getargs():
    parser = argparse.ArgumentParser(description='Controller micro-service')
    parser.add_argument('--port', '-p', required=False, dest='port', default='50050',
                        help='Port to start the controller on')
    parser.add_argument('--allocation_policy', '-ap', required=True,
                        dest='allocationPolicy', choices=['1', '2', '3', '4', '5', '6', '7', '8'],
                        help=(f'Allocation policy for the controller. 1: Round Robin, '
                              f'2: Tail-Heavy, 3: Cost-based ILP, 4: InferLine, 5: CascadeIteration'))
    parser.add_argument('--cascade', '-c', required=True,
                       dest='cascadeExec', choices=['sdturbo', 'sdxs', 'sdxlltn'],
                       help=(f'The cascade pipeline to execute.'))

    return parser.parse_args()


def serve(args):
    port = args.port
    allocationPolicy = AllocationPolicy(int(args.allocationPolicy))
    set_cas_exec(args.cascadeExec)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    controller = Controller(allocationPolicy=allocationPolicy)
    controller_pb2_grpc.add_ControllerServicer_to_server(controller, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logging.info(f'Controller started, listening on port {port}...')
    logging.info(f'Using resource allocation policy {allocationPolicy}')
    server.wait_for_termination()


if __name__=='__main__':
    logfile_name = f'../../logs/controller_{time.time()}.log'
    logging.basicConfig(filename=logfile_name, level=logging.INFO, 
                        format='%(asctime)s %(levelname)-8s %(message)s')
    serve(getargs())
