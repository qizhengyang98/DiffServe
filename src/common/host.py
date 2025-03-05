import grpc
from enum import Enum
from typing import List


class RoutingPolicy(Enum):
    EQUAL = 1
    MOST_ACCURATE_FIRST = 2

class RoutingEntry:
    def __init__(self, hostID: str, ip: str, port: str, task: str,
                 infer_level: int, percentage: float):
        self.hostID = hostID
        self.ip = ip
        self.port = port
        self.task = task
        self.infer_level = infer_level
        self.percentage = percentage

def getRoutingTableStr(routingTable):
    routingEntries = []
    for entry in routingTable:
        entryString = (f'hostID: {entry.hostID}, ip: {entry.ip}, port: {entry.port}, '
                       f'task: {entry.task}, percentage: {entry.percentage}')
        routingEntries.append(entryString)
    routingTableStr = f"[{','.join(routingEntries)}]"
    return routingTableStr

class Host:
    def __init__(self, hostID: str, ip:str, port: str, appID: str,
                 connection: grpc.Channel, routingTable: List[RoutingEntry]=[]):
        self.hostID = hostID
        self.ip = ip
        self.port = port
        self.appID = appID
        self.connection = connection
        self.routingTable = routingTable

    def addRoutingEntry(self, routingEntry):
        self.routingTable.append(routingEntry)

    def clearRoutingTable(self):
        oldTable = self.routingTable
        self.routingTable = []
        return oldTable
    

class Client(Host):
    # Only inherit the fields and methods of Host class
    pass

# We can add more fields for Worker as and when required by other load
# balancing policies
class Worker(Host):
    def __init__(self, hostID: str, ip: str, port: str, appID: str,
                 connection: grpc.Channel, task: str, loadedModel: str,
                 load: float, infer_level: int, routingTable: List[RoutingEntry]=[]):
        self.hostID = hostID
        self.ip = ip
        self.port = port
        self.appID = appID
        self.connection = connection
        self.task = task
        self.loadedModel = loadedModel
        self.load = load
        self.routingTable = routingTable
        self.infer_level = infer_level  # 0 for 1st level (small model) and 1 for 2nd level (large model)

        # addRoutingTable() and clearRoutingTable() are inherited from Host class
