import os
import sys
sys.path.append('..')
import argparse
import grpc
import logging
import pickle
import random
import threading
import time
import uuid
from array import array
from concurrent import futures
from multiprocessing import Pipe
import torch
import torch.multiprocessing as mp
from torch import tensor
from model import LoadedModel
from common.host import getRoutingTableStr
from protos import worker_pb2, worker_pb2_grpc
from protos import controller_pb2, controller_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from config import set_cas_exec, set_do_simulate_true


lock = threading.Lock()

class WorkerDaemon(worker_pb2_grpc.WorkerDaemonServicer):
    def __init__(self, workerIP: str, workerPort: str, controllerIP: str,
                 controllerPort: str):
        self.hostID = str(uuid.uuid4())

        self.appID = ''
        self.task = None
        self.childrenTasks = []

        # Model executor and preprocessing function
        self.preprocess = None
        childReadPipe, self.writePipe = Pipe()
        self.readPipe, childWritePipe = Pipe()
        self.loadedModel = LoadedModel((childReadPipe, self.writePipe), 
                                       (self.readPipe, childWritePipe))
        
        self.IP = workerIP
        self.port = workerPort

        self.controllerIP = controllerIP
        self.controllerPort = controllerPort

        self.lbIP = None
        self.lbPort = None

        self.controllerConnection = None
        self.lbConnection = None

        self.routingTable = []
        self.connections = {}
        self.queryMetaStore = {}
        
        self.modelName = None
        self.infer_level = None
        self.conf_thres = None
        self.batch_size = None
        self.onCUDA = int(torch.cuda.is_available())

        self.stats = {'queries_received': 0, 'queries_since_heartbeat': 0,
                      'queue_size': 0, 'branching': {}, 'branching_since_heartbeat': {}, 
                      'queries_timestamp_since_hearbeat': {}}

        # Setting up controller
        self.setupController(self.controllerIP, self.controllerPort)
        
        # Starting worker event loop
        eventLoopThread = threading.Thread(target=self.eventLoop)
        eventLoopThread.start()
        return
    

    def Heartbeat(self, request, context):
        message = f'Host {self.hostID} is still alive!'

        queriesSinceHeartbeat = self.stats['queries_since_heartbeat']
        totalQueries = self.stats['queries_received']
        queueSize = self.stats['queue_size']
        branchingSinceHeartbeat = pickle.dumps(self.stats['branching_since_heartbeat'])
        queriesTimestampSinceHearbeat = pickle.dumps(self.stats['queries_timestamp_since_hearbeat'])

        # Reset some stats at every heartbeat
        self.stats['queries_since_heartbeat'] = 0
        self.stats['branching_since_heartbeat'] = {}
        self.stats['queries_timestamp_since_hearbeat'] = {}

        return worker_pb2.HeartbeatResponse(message=message,
                                            queriesSinceHeartbeat=queriesSinceHeartbeat,
                                            totalQueries=totalQueries,
                                            queueSize=queueSize,
                                            onCUDA=self.onCUDA,
                                            branchingSinceHeartbeat=branchingSinceHeartbeat,
                                            queriesTimestampSinceHearbeat=queriesTimestampSinceHearbeat)
    

    def setupController(self, IP, port):
        logging.info('Setting up controller..')
        try:
            connection = grpc.insecure_channel(f'{IP}:{port}')
            stub = controller_pb2_grpc.ControllerStub(connection)
            request = controller_pb2.RegisterWorker(hostID=self.hostID, hostIP=self.IP,
                                                   hostPort=self.port)
            response = stub.WorkerSetup(request)

            self.controllerConnection = connection

            # If load balancer resides on same host as controller, use the same
            # IP for it
            if response.lbIP == 'localhost':
                self.lbIP = self.controllerIP
            else:
                self.lbIP = response.lbIP
                
            self.lbPort = response.lbPort
            logging.info(f'Response from controller: {response}')
        except Exception as e:
            logging.exception(f'Could not connect to controller, exception: {e}')
        
        return
    
    def setupLoadBalancer(self):
        ''' This is done every time a new model is loaded
        '''
        logging.info('Setting up load balancer..')
        try:
            connection = grpc.insecure_channel(f'{self.lbIP}:{self.lbPort}')
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            request = load_balancer_pb2.RegisterWorkerAtLB(hostID=self.hostID,
                                                           hostIP=self.IP,
                                                           port=self.port,
                                                           appID=self.appID,
                                                           task=self.task,
                                                           loadedModel=self.modelName,
                                                           infer_level=self.infer_level)
            response = stub.WorkerSetup(request)
            
            self.lbConnection = connection
            logging.info(f'Response from load balancer: {response}')

        except Exception as e:
            logging.exception(f'Could not connect to load balancer, exception: {e}')
        
        return
    
    
    def LoadModel(self, request, context):
        modelName = request.modelName
        schedulerName = request.schedulerName
        appID = request.applicationID
        task = request.task
        childrenTasks = pickle.loads(request.childrenTasks)
        labelToChildrenTasks = pickle.loads(request.labelToChildrenTasks)
        self.infer_level = request.infer_level
        self.conf_thres = request.conf_thres
        self.batch_size = request.batch_size

        try:
            if self.modelName == modelName:
                with lock:
                    self.writePipe.send('UPDATE_THRES')
                    self.writePipe.send(f'{self.conf_thres},{self.batch_size}')
            else:
                with lock:
                    self.writePipe.send('LOAD_MODEL')
                    self.writePipe.send(f'{modelName},{schedulerName},{self.infer_level},{self.conf_thres},{self.batch_size},{appID},{task}')
                    self.writePipe.send(childrenTasks)
                    self.writePipe.send(labelToChildrenTasks)
            
            self.appID = appID
            self.task = task
            self.childrenTasks = childrenTasks

            response = worker_pb2.LoadModelEnum.LOAD_INITIATED
            return worker_pb2.LoadModelResponse(response=response)
            # else:
            #     raise Exception(f'Unknown message received: {message}')
        
        except Exception as e:
            # If model loading failed, respond with fail code and error message
            logging.error(f'Model loading failed with exception: {e}')
            response = worker_pb2.LoadModelEnum.LOAD_FAILED
            return worker_pb2.LoadModelResponse(response=response,
                                                loadingTimeInUSec=0,
                                                message=str(e))


    def InitiateRequest(self, request, context):
        # logging.info(f'Received initial request: {request}')
        logging.info("Check initiateRequest")

        # We need a requestID whether we can serve it or not
        requestID = str(uuid.uuid4())

        return self.serveQuery(request, context, requestID)
        
    
    def IntermediateRequest(self, request, context):
        # logging.info(f'Received intermediate request: {request}')
        logging.info("Check IntermediateRequest")

        requestID = request.requestID

        return self.serveQuery(request, context, requestID)
    

    def serveQuery(self, request, context, requestID):
        message = None

        # TODO: we are assuming self.appID is already set, but it needs to be set by
        #       someone (either controller or load balancer)
        if self.appID == request.applicationID:
            try:
                # We are assuming this is where the query's latency timer starts
                # and do not take into account the network delay for query reaching
                # this point
                timestamp = time.time()

                request.requestID = requestID
                # TODO: don't make queryID equals to requestID
                request.queryID = requestID
                
                # compute the processing time since last timestamp
                last_timestamp = request.timestamp
                request.processingTime = request.processingTime + timestamp - last_timestamp if last_timestamp else 0
                request.timestamp = timestamp
                
                # logging.info(f"check timestamp: {timestamp}, request.timestamp: {request.timestamp}, time: {time.time()}")
                # logging.info(f"processingTime: {request.processingTime}, timestamp-last: {request.timestamp-last_timestamp}, SLO: {request.latencySLOInUSec}")
                
                if request.requestID not in self.stats['queries_timestamp_since_hearbeat']:
                    self.stats['queries_timestamp_since_hearbeat'][request.requestID] = request.timestamp

                event = {'event': 'WORKER_RECEIVED_REQUEST',
                         'requestID': request.requestID, 'queryID': request.queryID,
                         'userID': request.userID, 'appID': request.applicationID,
                         'sequenceNum': request.sequenceNum, 'timestamp': timestamp, 
                         'processing time': request.processingTime}
                # logging.info(f'EVENT,{str(event)}')
                logging.info(f'EVENT: WORKER_RECEIVED_REQUEST')
                
                with lock:
                    logging.info("Check send REQUEST")
                    logging.info(f'Putting request in worker IPC pipe, time: {time.time()}')
                    self.writePipe.send('REQUEST')
                    self.writePipe.send(request)
                    logging.info(f'Done putting request in worker IPC pipe, time: {time.time()}')

                status = worker_pb2.RequestStatus.ACCEPTED

            except Exception as e:
                status = worker_pb2.RequestStatus.REQUEST_FAILED
                message = str(e)
                print(f'Request failed with exception: {e}')
        else:
            logging.warning(f'Request received for invalid application ID: '
                            f'{request.applicationID}, worker runs application ID: '
                            f'{self.appID}')
            status = worker_pb2.RequestStatus.INVALID_APPLICATION
        
        # Return ACK
        if message is None:
            return worker_pb2.InferenceRequestAck(requestID=requestID,
                                                  status=status)
        else:
            return worker_pb2.InferenceRequestAck(requestID=requestID,
                                                  status=status,
                                                  message=message)
    

    def eventLoop(self):
        # for aligning the log names of model and worker
        self.writePipe.send(f'{self.port}')
        logging.info(f'Worker daemon event loop waiting')
        while True:
            message = ''
            try:
                if self.readPipe.poll(1):
                    message = self.readPipe.recv()
                    logging.info(f'Message received by worker daemon eventLoop: {message}')
                else:
                    logging.warning(f"Worker: No data received within timeout [MESSAGE]")

                if message == 'LOAD_MODEL_RESPONSE':
                    message = self.readPipe.recv()
                    modelName, loadedFrom, loadingTime = message.split(',')
                    self.modelName = modelName
                    # Notify load balancer of change in model
                    self.setupLoadBalancer()
                    logging.info(f"Check LOAD_MODEL_RESPONSE")

                elif 'QUEUED_QUERY' in message:
                    query = self.readPipe.recv()
                    queueSize = int(message.split(',')[1])
                    
                    # # TODO: catch the bug without hard-coded
                    # if 'QUEUE_SIZE' in query:
                    #     queueSize = int(query.split(',')[1])
                    #     query = self.readPipe.recv()
                    
                    self.stats['queue_size'] = queueSize
                    logging.info(f"Check QUEUED_QUERY: {self.stats['queue_size']}")
                    
                    self.registerQuery(query)
                    logging.info(f"Check QUEUED_QUERY, query: {query}")

                elif 'QUEUE_SIZE' in message:
                    queueSize = int(message.split(',')[1])
                    self.stats['queue_size'] = queueSize
                    logging.info(f"Check QUEUE_SIZE: {self.stats['queue_size']}")
                    pass

                elif message == 'COMPLETED_INFERENCE':
                    while True:
                        message = self.readPipe.recv()
                        if message == 'DONE_SENDING':
                            logging.info('Message received by worker daemon eventLoop: DONE_SENDING')
                            break

                        logging.info(f'inference results received from worker at time {time.time()}')
                        # logging.info(f'serialized message length: {len(message)}')

                        # Forward intermediate query using routing table to find where to
                        # direct each query

                        # Perhaps we can have task ID to distinguish?
                        # This ID could be configured by controller
                        # Daemon configures executor with task IDs
                        # Executor sends results separated by task IDs to daemon

                        # query_result = message
                        logging.info(f'SequenceNum: {message.sequenceNum}, prompt: {message.prompt}, results qualified: {message.resultQualified}')
                        if message.resultQualified == 0:
                            infer_level = 1
                            byte_data = None
                        elif message.resultQualified == 1:
                            infer_level = 2
                            # byte_data = pickle.dumps(message.data)
                            byte_data = None
                        self.forwardIntermediateQuery(message.queryID, infer_level, message.prompt, byte_data)
                        self.stats['branching_since_heartbeat'][message.queryID] = 1

    #                     queryID, prompts, results, results_qualified = message.split(',')
    #                     logging.info(f'results: {results}, results qualified: {results_qualified}')
    #                     results = eval(results)

    #                     image_data = results.images
    #                     byte_data = pickle.dumps(image_data)
    #                     print(f'length of pickled tensor: {len(byte_data)}')
    #                     if int(results_qualified) == 0:
    #                         infer_level = 1
    #                     else:
    #                         infer_level = 2
    #                     # self.forwardIntermediateQuery(queryID, task, prompts, byte_data)
    #                     self.forwardIntermediateQuery(queryID, infer_level, prompts, byte_data)

                        # self.logBranching(results)
            except Exception as e:
                logging.error(f"Worker: Error receiving data - {e}")

    
    def logBranching(self, results: dict):
        branching = {}
        for task in results:
            tensor_data = results[task]
            branching[task] = tensor_data.shape[0]
        resultsShape = list(map(lambda x: x.shape[0], list(results.values())))

        for task in self.childrenTasks:
            if task not in self.stats['branching']:
                self.stats['branching'][task] = []

            if task not in self.stats['branching_since_heartbeat']:
                self.stats['branching_since_heartbeat'][task] = []
            
            if task in branching:
                self.stats['branching'][task].append(branching[task])
                self.stats['branching_since_heartbeat'][task].append(branching[task])
            else:
                self.stats['branching'][task].append(0)
                self.stats['branching_since_heartbeat'][task].append(0)

        print(f'\n\nresults shape: {resultsShape}')
        print(f'branching: {branching}')
        print(f'self.stats[branching]: {self.stats["branching"]}')
        print(f'self.stats[branching_since_heartbeat]: {self.stats["branching_since_heartbeat"]}\n\n')
        return

    
    def registerQuery(self, query):
        # logging.info(f'check query: {query}')
        self.queryMetaStore[str(query.queryID)] = query
        # self.queryMetaStore[str(query.requestID)] = query
        self.stats['queries_received'] += 1
        self.stats['queries_since_heartbeat'] += 1
        logging.info(f"Check registerQuery, queries_received: {self.stats['queries_received']}")
        return

    
    # def forwardIntermediateQuery(self, queryID, task, prompts, data):
    def forwardIntermediateQuery(self, queryID, infer_level, prompts, data):
        # TODO: a periodic cleanup thread can delete metadata for old queries
        #       that have completed execution, or they could be cleaned up when
        #       they get forwarded through this function

        queryMetadata = self.queryMetaStore[queryID]

        appID = queryMetadata.applicationID
        hostID = self.getHostID(appID, infer_level)
        
        if hostID is None:
            logging.warning(f'forwardIntermediateQuery(): getHostID returned None, '
                            f'cannot forward query')
            return

        logging.debug(f'forwardIntermediateQuery -- appID: {appID}, hostID: {hostID}, '
                     f'queryMetadata: {queryMetadata}, data: {data}')

        # Send query to intermediate worker
        try:
            connection = self.connections[hostID]
            stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            inference_request = worker_pb2.InferenceRequest(requestID=queryMetadata.requestID,
                                                            queryID=queryID,
                                                            userID=queryMetadata.userID,
                                                            applicationID=queryMetadata.applicationID,
                                                            prompt=prompts,
                                                            data=data,
                                                            latencySLOInUSec=queryMetadata.latencySLOInUSec,
                                                            sequenceNum=queryMetadata.sequenceNum,
                                                            timestamp=queryMetadata.timestamp,
                                                            processingTime=queryMetadata.processingTime)
            response = stub.IntermediateRequest(inference_request)

            event = {'event': 'WORKER_FORWARDED_QUERY',
                    'requestID': queryMetadata.requestID, 'queryID': queryID,
                    'userID': queryMetadata.userID, 'appID': queryMetadata.applicationID,
                    'task': self.task, 'sequenceNum': queryMetadata.sequenceNum,
                    'timestamp': time.time()}
            
            # logging.info(f'EVENT,{str(event)}')
            logging.info("EVENT: WORKER_FORWARDED_QUERY")
            
            # logging.info(f'Received response from worker for intermediate query: {response}')
            logging.info(f'Received response from worker for intermediate query')
        
        except Exception as e:
            logging.error(f'Error sending intermediate query (queryID: {queryID}, '
                          f'requestID: {queryMetadata.requestID}) to worker '
                          f'(hostID: {hostID}): {e}')
        return
    

    def SetRoutingTable(self, request, context):
        routingTable = pickle.loads(request.routingTable)
        logging.info(f'Setting routing table at worker: {getRoutingTableStr(routingTable)}')
        self.routingTable = routingTable
        response = worker_pb2.RoutingTableResponse(message='Routing table set successfully!')

        for routingEntry in routingTable:
            if routingEntry.hostID not in self.connections:
                try:
                    connection = grpc.insecure_channel(f'{routingEntry.ip}:{routingEntry.port}')
                    self.connections[routingEntry.hostID] = connection
                    logging.info(f'SetRoutingTable(): Established connection with new '
                                f'worker (hostID: {routingEntry.hostID}, IP: {routingEntry.ip}, '
                                f'port: {routingEntry.port})')
                except Exception as e:
                    logging.error(f'Could not establish connection with worker (hostID: '
                                  f'{routingEntry.hostID}, IP: {routingEntry.ip}, port: '
                                  f'{routingEntry.port}): {e}')
                
        # logging.warning(f'SetRoutingTable(): Should we remove old connections no longer '
        #                 f'in routing table? Or perhaps we may get them again. It depends '
        #                 f'on the overhead of keeping old connections open')
        return response
    

    # def getHostID(self, appID, task):
    def getHostID(self, appID, infer_level):
        ''' This function implements probability-based routing from a routing
            table lookup.
        '''
        # filteredWorkers = list(filter(lambda x: x.task == task, self.routingTable))
        filteredWorkers = list(filter(lambda x: x.infer_level==infer_level, self.routingTable))

        if len(filteredWorkers) == 0:
            logging.error(f'getHostID(): No hosts found for task: {task}')
            return None

        weights = list(map(lambda x: x.percentage, filteredWorkers))
        selected = random.choices(filteredWorkers, weights, k=1)

        hostID = selected[0].hostID

        logging.debug(f'\n\nfilteredWorkers: {filteredWorkers}')
        logging.debug(f'weights: {weights}')
        logging.debug(f'selected: {selected}, hostID: {hostID}')

        return hostID
    

def serve(args):
    workerIP = args.workerIP
    workerPort = args.workerPort
    controllerIP = args.controllerIP
    controllerPort = args.controllerPort
    set_cas_exec(args.cascadeExec)
    
    if args.do_simulate:
        set_do_simulate_true()
    
    # logfile_name = f'../../logs/worker_{time.time()}.log'
    logfile_name = f'../../logs/worker_{workerPort}.log'
    logging.basicConfig(filename=logfile_name, level=logging.INFO, 
                        format='%(asctime)s %(levelname)-8s %(message)s')
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=None))
    workerDaemon = WorkerDaemon(workerIP=workerIP, workerPort=workerPort,
                                controllerIP=controllerIP, controllerPort=controllerPort)
    worker_pb2_grpc.add_WorkerDaemonServicer_to_server(workerDaemon, server)
    server.add_insecure_port(f'[::]:{workerPort}')
    server.start()
    logging.info(f'Worker daemon started, listening on port {workerPort}...')
    server.wait_for_termination()


def getargs():
    parser = argparse.ArgumentParser(description='Worker daemon')
    parser.add_argument('--ip_address', '-ip', required=False, dest='workerIP',
                        default='localhost', help='IP address to start worker on')
    parser.add_argument('--port', '-p', required=False, dest='workerPort', default='50051',
                        help='Port to start worker on')
    parser.add_argument('--controller_ip', '-cip', required=False, dest='controllerIP',
                        default='localhost', help='IP address of the controller')
    parser.add_argument('--controller_port', '-cport', required=False,
                        dest='controllerPort', default='50050',
                        help='Port of the controller')
    parser.add_argument('--cascade', '-c', required=True,
                       dest='cascadeExec', choices=['sdturbo', 'sdxs', 'sdxlltn'],
                       help=(f'The cascade pipeline to execute.'))
    parser.add_argument('--do_simulate', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable workers simulate the execution of models (default: False)')

    return parser.parse_args()


if __name__=='__main__':
    mp.set_start_method('spawn', force=True)
    
    # logfile_name = f'../../logs/worker_{time.time()}.log'
    # logging.basicConfig(filename=logfile_name, level=logging.INFO, 
    #                     format='%(asctime)s %(levelname)-8s %(message)s')
    # asyncio.run(serve())
    serve(getargs())
