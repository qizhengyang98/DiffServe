# This is the Load Balancer microservice
import argparse
import sys
sys.path.append('..')
import grpc
import logging
import pickle
import time
import threading
from concurrent import futures
from common.app import App, AppNode, registerApplication
from common.host import Client, Worker, RoutingEntry, RoutingPolicy, getRoutingTableStr
from protos import client_pb2_grpc
from protos import controller_pb2, controller_pb2_grpc
from protos import load_balancer_pb2, load_balancer_pb2_grpc
from protos import worker_pb2, worker_pb2_grpc
from config import get_cas_exec, set_cas_exec


class LoadBalancer(load_balancer_pb2_grpc.LoadBalancerServicer):
    def __init__(self, port: str, controllerIP: str, controllerPort: str):
        self.IP = "localhost"
        self.port = port
        self.controllerIP = controllerIP
        self.controllerPort = controllerPort
        self.setupController(self.controllerIP, self.controllerPort)
        self.workers = []
        self.clients = []
        
        PIPELINE = get_cas_exec()
        logging.info(f"Current cascade pipeline: {PIPELINE}")
        # self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdturbo_sdv15.json')]
        if PIPELINE == 'sdturbo':
            self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdturbo_sdv15.json')]
        elif PIPELINE == 'sdxs':
            self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdxs_sdv15.json')]
        elif PIPELINE == 'sdxlltn':
            self.apps = [registerApplication('../../traces/apps/sdturbo_sdv15/sdxlltn_sdxl.json')]

        # TODO: make routing policy changeable
        self.routingPolicy = RoutingPolicy.EQUAL

        eventLoopThread = threading.Thread(target=self.eventLoop)
        eventLoopThread.start()


    def setupController(self, IP, port):
        print('Setting up controller..')
        try:
            connection = grpc.insecure_channel(f'{IP}:{port}')
            stub = controller_pb2_grpc.ControllerStub(connection)
            request = controller_pb2.RegisterLB(lbIP=self.IP, lbPort=self.port)
            response = stub.LBSetup(request)
            print(f'Response from controller: {response}')
        except Exception as e:
            logging.exception(f'Could not connect to controller, exception: {e}')
            raise e
        
        return
    

    def LBAlive(self, request, context):
        message = f'Load balancer is still alive!'
        return load_balancer_pb2.LBHeartbeatResponse(message=message)
    

    def WorkerSetup(self, request, context):
        ''' This is called every time a worker changes its loaded model
        '''
        logging.warning(f'Creating new entry for worker. If worker entry already '
                        f'exists, it should be replaced.')
        try:
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                workerIP = splitContext[1]
            else:
                workerIP = 'localhost'

            logging.info(f'Received request: {request}')

            connection = grpc.insecure_channel(f'{workerIP}:{request.port}')
            logging.info(f'GRPC connection established with worker (hostID: {request.hostID})')
            
            worker = Worker(hostID=request.hostID, ip=workerIP, port=request.port,
                            appID=request.appID, connection=connection, task=request.task,
                            loadedModel=request.loadedModel, load=0, infer_level=request.infer_level)
            # check if the work already exists in self.workers 
            workerExisted = [idx for idx, w in enumerate(self.workers) if w.hostID == worker.hostID]
            # if not, add the worker to the list
            if len(workerExisted) == 0:
                self.workers.append(worker)
            # if exists, update the information of the worker
            else:
                idx = workerExisted[0]
                self.workers[idx].task = request.task
                self.workers[idx].loadedModel = request.loadedModel
                self.workers[idx].infer_level = request.infer_level
            logging.info(f'Worker {request.hostID} added to list of available workers')

            return load_balancer_pb2.RegisterWorkerAtLBResponse(message='Done!')
        
        except Exception as e:
            raise e
    

    def ClientSetup(self, request, context):
        ''' Registers a new client
        '''
        logging.warning(f'Creating new entry for cleint. If client entry already '
                        f'exists, it should be replaced.')
        try:
            logging.info(f'context.peer(): {context.peer()}')
            splitContext = context.peer().split(':')
            if splitContext[0] == 'ipv4':
                clientIP = splitContext[1]
            else:
                clientIP = 'localhost'

            logging.info(f'Received request: {request}')

            connection = grpc.insecure_channel(f'{clientIP}:{request.port}')
            logging.info(f'GRPC connection established with client (hostID: {request.hostID})')
            
            client = Client(hostID=request.hostID, ip=clientIP, port=request.port,
                            appID=request.appID, connection=connection)
            self.clients.append(client)
            logging.info(f'Client {request.hostID} added to list of clients')

            return load_balancer_pb2.RegisterWorkerAtLBResponse(message='Done!')
        
        except Exception as e:
            raise e
    

    def balanceLoad(self):
        if self.routingPolicy == RoutingPolicy.EQUAL:
            self.balanceLoadEqual()
        else:
            raise Exception(f'Routing policy {self.routingPolicy} not implemented')
        
        # Once load is balanced, propagate the updated routing tables to the
        # clients and workers
        self.propagateRoutingTables()

        return
    

    def balanceLoadEqual(self):
        ''' Uses equal load balancing. For each client or worker, counts the number
        of workers at the next stage and divides requests amongst them equally.
        It is agnostic of the model variants loaded.
        '''
        logging.warning(f'This needs to be fixed. It has the same logical error'
                        f' that we fixed in the simulator. If a worker is '
                        f'pending removal but it is still processing its out-'
                        f'standing requests, we will not be including it in the '
                        f'routing table of other workers, but we will also stop '
                        f'updating its own routing table. Workers removed after '
                        f'it will still exist in its routing table and it will '
                        f'send requests to a potentially removed worker')
        for app in self.apps:
            print()
            print(f'Balancing load for app: {app}, appID: {app.appID}')
            app.print()
            print()
            clients = list(filter(lambda x: x.appID == app.appID, self.clients))
            workers = list(filter(lambda x: x.appID == app.appID, self.workers))

            # TODO: we are doing the same thing for both clients and workers
            #       it would be better to use a function
            for client in clients:
                oldTable = client.clearRoutingTable()

                child = app.root
                filteredWorkers = list(filter(lambda x: x.task == child.task,
                                              self.workers))
                filteredWorkers = list(filter(lambda x: x.infer_level == 0, filteredWorkers))
                
                if len(filteredWorkers) == 0:
                    filteredWorkers = list(filter(lambda x: x.infer_level == 1, self.workers)) # all queries go to 2nd-level worker
                    if len(filteredWorkers) == 0:
                        logging.error(f'No workers found for task: {child.task} or heavy model')
                        client.routingTable = oldTable
                        continue

                for filteredWorker in filteredWorkers:
                    routingEntry = RoutingEntry(hostID=filteredWorker.hostID,
                                                ip=filteredWorker.ip,
                                                port=filteredWorker.port,
                                                task=filteredWorker.task,
                                                infer_level=filteredWorker.infer_level,
                                                percentage=1/len(filteredWorkers))
                    client.addRoutingEntry(routingEntry)
                
                logging.info("client -----------------------")
                logging.info(f'Client {client.hostID} routing table: '
                             f'{getRoutingTableStr(client.routingTable)}')
                    
            for worker in workers:
                logging.info(f"Worker info: {worker.hostID}, {worker.infer_level}, {worker.task}")
                # Clear the routing table for now. If no new table is constructed,
                # re-use the old table
                oldTable = worker.clearRoutingTable()
                newTableConstructed = False

                # Find appNode in the app graph
                node = app.findNodeByHost(worker)

                if node is None:
                    raise Exception('app.findNodeByHost returned None')
                if len(node.children) == 0:
                    logging.warning(f'Encountered node with no outgoing edges: {node.task},'
                                    f' do not know how to handle it yet')

                for child in node.children:
                    if worker.infer_level == 0:
                        filteredWorkers = list(filter(lambda x: x.infer_level >= 1, self.workers))
                    elif worker.infer_level == 1:
                        filteredWorkers = list(filter(lambda x: x.infer_level == 2, self.workers))
                    if len(filteredWorkers) == 0:
                        logging.warning(f'No workers found for heavy model or sink')
                        continue

                    for filteredWorker in filteredWorkers:
                        routingEntry = RoutingEntry(hostID=filteredWorker.hostID,
                                                    ip=filteredWorker.ip,
                                                    port=filteredWorker.port,
                                                    task=filteredWorker.task,
                                                    infer_level=filteredWorker.infer_level,
                                                    percentage=1/len(filteredWorkers))
                        worker.addRoutingEntry(routingEntry)

                    newTableConstructed = True
                
                # Re-use the old table if no new table is constructed
                if not(newTableConstructed):
                    worker.routingTable = oldTable
                
                logging.info(f'Worker {worker.hostID} routing table: '
                             f'{getRoutingTableStr(worker.routingTable)}')

        return
    

    def propagateRoutingTables(self):
        for client in self.clients:
            self.propagateRoutingTableToHost(client)
        for worker in self.workers:
            self.propagateRoutingTableToHost(worker)
        return
    

    def propagateRoutingTableToHost(self, host):
        ''' Sends constructed routing table to host (client or worker)
        '''
        try:
            pickledTable = pickle.dumps(host.routingTable)

            connection = host.connection
            hostType = type(host).__name__
            if hostType == 'Client':
                stub = client_pb2_grpc.ClientStub(connection)
            elif hostType == 'Worker':
                stub = worker_pb2_grpc.WorkerDaemonStub(connection)
            else:
                raise Exception(f'Unknown hostType encountered: {hostType}')

            request = worker_pb2.RoutingTableRequest(routingTable=pickledTable)
            response = stub.SetRoutingTable(request)
            logging.info(f'SetRoutingTable response from host (hostID: {host.hostID}): '
                        f'{response.message}')
        
        except Exception as e:
            logging.warning(f'Could not contact host to propagate routing table '
                            f'(hostID: {host.hostID}): {e}')
    

    def eventLoop(self):
        while True:
            logging.info('Event loop woke up from sleep. Balancing load..')
            workerModels = list(map(lambda x: x.loadedModel, self.workers))
            logging.info(f'List of available models loaded on workers: {workerModels}')

            self.balanceLoad()

            time.sleep(1)


def getargs():
    parser = argparse.ArgumentParser(description='Load balancer micro-service')
    parser.add_argument('--port', '-p', required=False, dest='port', default='50048',
                        help='Port to start load balancer on')
    parser.add_argument('--controller_ip', '-cip', required=False, dest='controllerIP',
                        default='localhost', help='IP address of the controller')
    parser.add_argument('--controller_port', '-cport', required=False,
                        dest='controllerPort', default='50050',
                        help='Port of the controller')
    parser.add_argument('--cascade', '-c', required=True,
                       dest='cascadeExec', choices=['sdturbo', 'sdxs', 'sdxlltn'],
                       help=(f'The cascade pipeline to execute.'))

    return parser.parse_args()
    

def serve(args):
    port = args.port
    controllerIP = args.controllerIP
    controllerPort = args.controllerPort
    set_cas_exec(args.cascadeExec)
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    loadBalancer = LoadBalancer(port=port, controllerIP=controllerIP,
                                controllerPort=controllerPort)
    load_balancer_pb2_grpc.add_LoadBalancerServicer_to_server(loadBalancer, server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f'Load balancer started, listening on port {port}...')
    server.wait_for_termination()


if __name__=='__main__':
    logfile_name = f'../../logs/load_balancer_{time.time()}.log'
    logging.basicConfig(filename=logfile_name, level=logging.INFO, 
                        format='%(asctime)s %(levelname)-8s %(message)s')
    serve(getargs())
