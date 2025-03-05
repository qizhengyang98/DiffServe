import argparse
import sys
sys.path.append('..')
import grpc
import pickle
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from concurrent import futures
from common.host import getRoutingTableStr
from protos import client_pb2, client_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from protos import worker_pb2, worker_pb2_grpc


MSECONDS_IN_SEC = 1000

class ClientDaemon(client_pb2_grpc.ClientServicer):
    def __init__(self, ip: str, port: str, lbIP: str, lbPort: str, dataPath: Path, tracePath: Path):
        self.hostID = str(uuid.uuid4())
        logging.info(f'Client started with ID: {self.hostID}')

        # TODO: replace hard-coded appID
        self.appID = '131'

        self.routingTable = []
        self.connections = {}

        self.ip = ip
        self.port = port

        self.lbIP = lbIP
        self.lbPort = lbPort
        self.lbConnection = None

        self.setupLB()

        self.datasets = self.prompt_load(dataPath)
        self.traces = self.trace_load(tracePath)
        # sendRequestsThread = threading.Thread(target=self.sendRequests)
        sendRequestsThread = threading.Thread(target=self.sendRequestsByTraces)
        sendRequestsThread.start()
        
    def prompt_load(self, input_prompt: Path, limit_prompts=None):
        prompts = None
        if input_prompt is not None:  # by file
            with open(input_prompt, encoding='ascii', errors='ignore') as f:
                prompts = f.read().splitlines()
            if limit_prompts is not None:
                prompts = prompts[0 : limit_prompts]
        else:  # by txt/list
            pass
        return prompts
    
    def trace_load(self, input_trace: Path):
        requests_trace = []
        if input_trace is not None: # by file
            with open(input_trace, encoding='ascii', errors='ignore') as f:
                str_requests = f.read().splitlines()
        else:
            pass
        for req in str_requests:
            in_timestamp, ind_prompt = req.split(',') 
            in_timestamp = float(in_timestamp) / MSECONDS_IN_SEC # request coming time in second
            ind_prompt = int(ind_prompt) # index of prompt in a dataset
            requests_trace.append((in_timestamp, ind_prompt))
        requests_trace.sort()
        return requests_trace
    
    
    def setupLB(self):
        logging.info('Setting up load balancer at client..')
        try:
            connection = grpc.insecure_channel(f'{self.lbIP}:{self.lbPort}')
            stub = load_balancer_pb2_grpc.LoadBalancerStub(connection)
            request = load_balancer_pb2.RegisterClient(hostID=self.hostID,
                                                       hostIP=self.ip,
                                                       port=self.port,
                                                       appID=self.appID)
            response = stub.ClientSetup(request)
            
            self.lbConnection = connection
            logging.info(f'Response from load balancer: {response}')

        except Exception as e:
            logging.exception(f'Could not connect to load balancer, exception: {e}')


    def SetRoutingTable(self, request, context):
        routingTable = pickle.loads(request.routingTable)
        logging.info(f'Setting routing table at client: {getRoutingTableStr(routingTable)}')
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
                
        # TODO: fix this warning
        logging.warning(f'SetRoutingTable(): Should we remove old connections no longer '
                        f'in routing table? Or perhaps we may get them again. It depends '
                        f'on the overhead of keeping old connections open')
        
        # TODO: fix this warning
        # logging.warning(f'SetRoutingTable() at client has the same code as on worker, '
        #                 f'so a change in one may not reflect in both places. Perhaps '
        #                 f'they should both inherit from a Host class')
        return response
    

    def getHostID(self):
        # TODO: create a host daemon class and inherit this from it
        ''' This function will implement probability-based routing from a routing
            table lookup.
        '''
        task = self.routingTable[0].task

        filteredWorkers = list(filter(lambda x: x.task == task, self.routingTable))

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
    
    
    def sendRequestsByTraces(self):
        latencySLOInUSec = 15 * 1000 * 1000
        logging.info('Sleeping for 10 seconds before sending requests')
        time.sleep(10)
        
        logging.info('Start sending requests')
        request_start_time = time.time()
        last_request_timestamp = 0
        for in_timestamp, data_idx in self.traces:
            waiting_time = in_timestamp - last_request_timestamp
            time_difference = time.time() - request_start_time
            sleeping_time = waiting_time-time_difference if waiting_time>=time_difference else 0
            logging.info(f'Request sent at timestamp {in_timestamp}, sleeping for {sleeping_time} before sending the request...')
            print(f'Request sent at timestamp {in_timestamp}, sleeping for {sleeping_time} before sending the request...')
            time.sleep(sleeping_time)
            
            prompt = self.datasets[data_idx]
            
            request_start_time = time.time()
            last_request_timestamp = in_timestamp
            
            inference_request = worker_pb2.InferenceRequest(userID=self.hostID,
                                                            applicationID='131',
                                                            prompt=prompt,
                                                            latencySLOInUSec=latencySLOInUSec,
                                                            sequenceNum=data_idx)
            hostID = self.getHostID()
            
            try:
                connection = self.connections[hostID]
                stub = worker_pb2_grpc.WorkerDaemonStub(connection)

                logging.info(f'Sending request (sequenceNum: {data_idx}) to '
                            f'worker (hostID: {hostID})..')
                
                response = stub.InitiateRequest(inference_request)
                logging.info(f'Message from worker for inference request number '
                            f'{data_idx}.. requestID: {response.requestID}, request '
                            f'status: {response.status}, message: {response.message}')
                
            except Exception as e:
                logging.warning(f'Could not send request (sequenceNum: {data_idx}) '
                                f'to worker (hostID: {hostID}): {e}')
                
        logging.info(f"Trace ended, stop sending requests.")
        print(f"Trace ended, stop sending requests.")
    

    def sendRequests(self):
        logging.info('Sleeping for 15 seconds before sending requests..')
        time.sleep(15)

        latencySLOInUSec = 5 * 1000 * 1000
        request_start_time = time.time()
        for data_idx in range(5000):
            logging.info('Sleeping for 250 milliseconds before sending another request..')
            time_difference = time.time() - request_start_time
            time.sleep(0.25 - time_difference)

            logging.info(f'Time difference was: {time_difference}')
            prompt = self.datasets[data_idx]

            request_start_time = time.time()
            inference_request = worker_pb2.InferenceRequest(userID=self.hostID,
                                                            applicationID='131',
                                                            prompt=prompt,
                                                            latencySLOInUSec=latencySLOInUSec,
                                                            sequenceNum=data_idx)
            hostID = self.getHostID()

            try:
                connection = self.connections[hostID]
                stub = worker_pb2_grpc.WorkerDaemonStub(connection)

                logging.info(f'Sending request (sequenceNum: {data_idx}) to '
                            f'worker (hostID: {hostID})..')
                
                response = stub.InitiateRequest(inference_request)
                logging.info(f'Message from worker for inference request number '
                            f'{data_idx}.. requestID: {response.requestID}, request '
                            f'status: {response.status}, message: {response.message}')
                
            except Exception as e:
                logging.warning(f'Could not send request (sequenceNum: {data_idx}) '
                                f'to worker (hostID: {hostID}): {e}')
                

    def add_requests_from_trace_pointer(self, isi_name, readfile, read_until=500):
        ''' Partially read trace file into memory and store pointer.
        '''
        while True:
            line = readfile.readline()
            if not line:
                return True
            if line.strip()[0] == '#':
                # Skip comment lines in file
                continue
            request_description = line.rstrip('\n').split(',')
            start_time = int(float(request_description[0]))
            # if qos_level is defined, use that. otherwise use qos_level 0 by default
            if len(request_description) >= 2:
                try:
                    qos_level = int(request_description[1])
                except ValueError:
                    accuracy = float(request_description[1])
                    qos_level = 0
            else:
                qos_level = 0
            # if n_qos_levels == 1, ignore qos info from trace
            if self.n_qos_levels == 1:
                qos_level = 0

            deadline = self.slo_dict[isi_name]

            if len(request_description) >= 4:
                accuracy = float(request_description[3])
            else:
                accuracy = 100.0

            app_name = self.root_isi_to_app_name[isi_name]

            # The path is assigned when the Executor assigns the request
            # to a Predictor
            _, event = self.insert_event(start_time, EventType.START_REQUEST, isi_name,
                                      runtime=None, deadline=deadline,
                                      qos_level=qos_level, accuracy=accuracy,
                                      sequence_num=self.parent_requests_added,
                                      path=None, app_name=app_name,
                                      app_parent_task=isi_name)
            self.requests_added += 1
            self.parent_requests_added += 1

            # We want to keep track of the arrival times of all parent requests
            parent_request_id = event.parent_request_id
            if parent_request_id == '' or parent_request_id is None:
                raise Exception(f'Event was not assigned a parent_request_id')
            self.parent_request_arrival_times[parent_request_id] = start_time

            if start_time >= read_until:
                break

            self.last_request_starting[isi_name] = start_time
        return False


def serve(args):
    ip = args.ip
    port = args.port
    lbIP = args.lbIP
    lbPort = args.lbPort
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    textpath = "../../traces/text_prompt_mscoco_120k.txt"
    tracePath = f"../../traces/maf/sdturbo_sdv15_dynamic_testbed/trace_{args.trace}qps.txt"
    # tracePath = "../../traces/maf/sdturbo_sdv15_dynamic_testbed/trace_4to32qps.txt" # 2to16, 2_5to20, 3to24, 3_5to28, 4to32, 3to40, 3_5to40, 1to8, 2to10
    client = ClientDaemon(ip=ip, port=port, lbIP=lbIP, lbPort=lbPort, dataPath=textpath, tracePath=tracePath)
    client_pb2_grpc.add_ClientServicer_to_server(client, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f'Client started, listening on port {port}...')
    server.wait_for_termination()


def getargs():
    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--ip_address', '-ip', required=False, dest='ip',
                        default='localhost', help='IP address to start client on')
    parser.add_argument('--port', '-p', required=False, dest='port', default='51050',
                        help='Port to start client on')
    parser.add_argument('--lb_ip', '-lbip', required=False, dest='lbIP',
                        default='localhost', help='IP address of the load balancer')
    parser.add_argument('--lb_port', '-lbport', required=False, dest='lbPort',
                        default='50048', help='Port of the load balancer')
    parser.add_argument('--trace_file', '-trace', required=False, dest='trace',
                        default='4to32', help='The trace file used for client')

    return parser.parse_args()


if __name__=='__main__':
    logfile_name = f'../../logs/client_{time.time()}.log'
    logging.basicConfig(filename=logfile_name, level=logging.INFO, 
                        format='%(asctime)s %(levelname)-8s %(message)s')
    # client = Client()
    # client.run()
    # print('Done')

    serve(getargs())
