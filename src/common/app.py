import json
import logging
import pandas as pd
from collections import OrderedDict
from typing import List
from common.host import Host


USecInMSec = 1000
MSecInSec = 1000 
USecInSec = MSecInSec * USecInMSec

class AppNode:
    def __init__(self, task: str, label: str, modelVariants: List[str],
                 children=[]):
        self.task = task
        self.label = label
        self.modelVariants = modelVariants
        self.children = children

        self.parents = []

        self.maxLatency = None
        self.throughput = None
        self.incomingThroughput = None
        self.height = None
    
    def getTaskName(self) -> str:
        return self.task
    
    def addChildren(self, children=[]):
        for child in children:
            self.children.append(child)
        return

    def addChild(self, child):
        self.children.append(child)
        return


class App:
    def __init__(self, root: AppNode, appID: str, appName: str, latencySLOInMSec: int):
        self.root = root
        self.appID = appID
        self.appName = appName
        self.latencySLOInUSec = latencySLOInMSec * USecInMSec

    def getName(self) -> str:
        return self.appName

    def getLatencySLO(self) -> int:
        return self.latencySLOInUSec
    
    def getRootNode(self) -> AppNode:
        return self.root

    def findNodeByHost(self, host: Host) -> AppNode:
        ''' Start at root and perform graph search to get node matching task of
            host
        '''
        return self.findNodeByHostHelper(self.root, host)
    
    def findNodeByHostHelper(self, node: AppNode, host: Host) -> AppNode:
        if node.task == host.task:
            return node
        else:
            for child in node.children:
                foundInChild = self.findNodeByHostHelper(child, host)
                if foundInChild is not None:
                    return foundInChild
            return None
    
    def findNodeByTask(self, task: str) -> AppNode:
        return findNodeByTaskHelper(self.root, task)
    
    def findTaskFromModelVariant(self, modelVariant: str) -> str:
        ''' Given a model variant name as input, searches through the App
            tree to find which task it belongs to
        '''
        return self.findTaskFromModelVariantHelper(self.root, modelVariant)
    
    def findTaskFromModelVariantHelper(self, node: AppNode, modelVariant: str) -> str:
        if modelVariant in node.modelVariants:
            return node.task
        
        for child in node.children:
            foundInChild = self.findTaskFromModelVariantHelper(child, modelVariant)
            if foundInChild is not None:
                return foundInChild
        
        return None
        
    def getNodeString(self, node: AppNode) -> str:
        ''' Construct a string representation of the subtree below a given node
        '''
        childrenStrings = []
        for child in node.children:
            childString = self.getNodeString(child)
            childrenStrings.append(childString)
        childrenString = ','.join(childrenStrings)

        parentTasks = list(map(lambda x: x.task, node.parents))
        parentTasksString = ','.join(parentTasks)

        nodeString = (f'(task: {node.task}, label: {node.label}, model variants: '
                      f'{node.modelVariants}, children: [{childrenString}], '
                      f'parents task names: {parentTasksString})')

        return nodeString
    
    def getAllTasks(self) -> List[str]:
        ''' Returns the names of all tasks in the App as a list
        '''
        # return self.getAllTasksHelper(self.root)
        allNodes = self.getAllNodesHelper(self.root)
        allTasks = list(map(lambda x: x.task, allNodes))
        return allTasks
    
    def getAllNodesHelper(self, node: AppNode) -> List[AppNode]:
        allChildrenNodes = [node]
        
        for child in node.children:
            childNodes = self.getAllNodesHelper(child)

            for childNode in childNodes:
                if childNode not in allChildrenNodes:
                    allChildrenNodes.append(childNode)
        
        return allChildrenNodes
    
    def getAllTasksTopSorted(self) -> List[str]:
        ''' Returns the names of all tasks in the App as a list, sorted in
            topological order
        '''
        # BFS would not work on the social media graph
        # We need to sort nodes in order of their heights

        self.setHeights(self.root)

        # We will iterate through the graph and create a dict mapping
        # AppNode -> height
        toVisitNodes = [self.root]
        visitedDict = {}
        while len(toVisitNodes) > 0:
            visiting = toVisitNodes.pop(0)
            if visiting.height is None:
                raise Exception('Node has height None')
            
            visitedDict[visiting] = visiting.height

            # Add all children
            for child in visiting.children:
                toVisitNodes.append(child)
        
        # We now sort the dict in reverse order of height
        sortedDict = OrderedDict(sorted(visitedDict.items(),
                                        key=lambda item: item[1],
                                        reverse=True))
        
        # Create the list of topologically sorted tasks
        topologicalSorted = []
        for node in sortedDict:
            topologicalSorted.append(node.task)

        return topologicalSorted
    
    def setHeights(self, node: AppNode):
        ''' Visits each node in the subgraph and sets its height
        '''
        maxChildHeight = 0
        for child in node.children:
            childHeight = self.setHeights(child)
            maxChildHeight = max(childHeight, maxChildHeight)

        node.height = maxChildHeight + 1
        return node.height
    
    def getModelVariantsFromTaskName(self, taskName: str) -> List[str]:
        ''' For a given task name, return the list of all its model variants
        '''
        node = self.findNodeByTask(taskName)
        return node.modelVariants
    
    def getAllTasksHelper(self, node: AppNode) -> List[str]:
        allChildrenTasks = [node.task]

        for child in node.children:
            childTasks = self.getAllTasksHelper(child)
            allChildrenTasks.extend(childTasks)

        return allChildrenTasks

    def getChildrenTasks(self, task: str) -> List[str]:
        ''' Find the AppNode corresponding to the given task and construct a
            list of task names for all its children
        '''
        childrenTasks = []

        node = self.findNodeByTask(task)
        if node is not None:
            childrenTasks = list(map(lambda x: x.task, node.children))

        return childrenTasks
    
    def getLabelToChildrenTasksDict(self, task: str) -> dict:
        ''' Find the AppNode corresponding to the given task and construct a
            dict with (key: label, value: task name) for all its children
        '''
        labelToTasksDict = {}

        node = self.findNodeByTask(task)
        if node is None:
            return labelToTasksDict
        
        for child in node.children:
            labelToTasksDict[child.label] = child.task

        return labelToTasksDict
    
    def findMinThroughput(self, allocationPlan: dict, executionProfiles: pd.DataFrame,
                          branchingProfiles: pd.DataFrame) -> str:
        ''' Given an allocation plan, execution profiles, and branching profiles,
            returns the task that has the minimum throughput in the plan
        '''
        # This is what we need to do
        # 1. We first create a mirror graph of the app with each node/task
        #    representing the throughput of each task
        self.setThroughputsFromAllocation(allocationPlan=allocationPlan,
                                          executionProfiles=executionProfiles,
                                          branchingProfiles=branchingProfiles)

        # # 2. Then we trace through the graph and find the task with the smallest
        # #    throughput, considering branching profiles]
        (gap, task) = self.findThroughputBottleneck(self.root)
        
        print(f'\n\nFinal gap returned: {gap}, task: {task}\n\n')
    
        self.resetThroughputs(node=self.root)
        return task, thput
    
    def findAllocThroughput(self, allocationPlan: dict):
        '''
        Given a allocation plan, return the throughput for both 1st and 2nd level workers
        '''
        # TODO: return the throughput of the few-step and normal SD model directly
        return thput1st, thput2nd

    def findThroughputBottleneck(self, node: AppNode) -> str:
        raise Exception(f'This will break because instead of a single parent, '
                        f'we can have multiple parents in a DAG')
        throughput = node.throughput
        incomingThroughput = node.incomingThroughput
        task = node.task

        parentTask = None
        if node.parent is not None:
            parentTask = node.parent.task

        print(f'task: {task}, parentTask: {parentTask}, throughput: {throughput}, '
              f'incomingThroughput: {incomingThroughput}')

        if len(node.children) == 0:
            return (0, None)
        if node.parent is None:
            gap = 0
        else:
            gap = incomingThroughput - throughput

        maxChildGap = 0
        maxGapTask = None
        for child in node.children:
            (childGap, task) = self.findThroughputBottleneck(node=child)
            if childGap > maxChildGap:
                maxChildGap = childGap
                maxGapTask = task
        
        if abs(gap) > maxChildGap:
            if gap > 0:
                print(f'Returning gap: {gap}, task: {node.task}')
                return gap, node.task
            else:
                print(f'Returning gap: {abs(gap)}, task: {node.parent.task}')
                return abs(gap), node.parent.task
        else:
            print(f'Returning gap: {maxChildGap}, task: {maxGapTask}')
            return (maxChildGap, maxGapTask)
    
    def setThroughputsFromAllocation(self, allocationPlan: dict,
                                     executionProfiles: pd.DataFrame,
                                     branchingProfiles: pd.DataFrame):
        ''' Given an allocation plan and execution profile, sets the total
            throughput for each task in self's application graph
        '''
        for key in allocationPlan:
            (modelVariant, batchSize, hardware) = key
            replicas = allocationPlan[key]

            # ----------------
            # Setting node throughputs using allocationPlan
            # ----------------
            row = executionProfiles.loc[(executionProfiles['Model'].str.contains(modelVariant)) &
                                        (executionProfiles['batchsize'] == batchSize) &
                                        (executionProfiles['Accel'] == hardware)]

            if len(row) > 0:
                latencyInSec = float(row['90th_pct'].values[0])
                throughput = replicas * batchSize / latencyInSec
            else:
                # TODO: what to do if profiled data is not found?
                latencyInSec = 0
                throughput = 0
                logging.error(f'findMinThroughput(): profiled data not found for '
                              f'key: {key}')

            task = self.findTaskFromModelVariant(modelVariant=modelVariant)
            node = self.findNodeByTask(task=task)
            self.addThroughput(node=node, throughput=throughput)

            # ----------------
            # Setting parent throughputs using branching profiles
            # ----------------

            branchingRow = branchingProfiles.loc[branchingProfiles['model'].str.contains(modelVariant)]

            print('\n')
            if len(branchingRow) > 0:
                multFactor = float(branchingRow['mult_factor'].values[0])

                for childTask in branchingRow:
                    if childTask == 'model' or childTask == 'mult_factor' or 'Unnamed' in childTask:
                        continue
                    childBranchingFactor = branchingRow[childTask].values[0]
                    incomingThroughput = throughput * multFactor * childBranchingFactor
                    childNode = self.findNodeByTask(childTask)

                    print(f'\nmodelVariant: {modelVariant}, childTask: {childTask}, '
                          f'childBranchingFactor: {childBranchingFactor}, '
                          f'parentRawThroughput: {throughput}, parentCalculatedThroughput: '
                          f'{incomingThroughput}, childNode.task: {childNode.task}')
                    
                    self.addincomingThroughput(node=childNode,
                                             incomingThroughput=incomingThroughput)

            else:
                multFactor = 1
                for child in node.children:
                    self.addincomingThroughput(child, incomingThroughput=throughput)

            print(f'multFactor: {multFactor}')

        return
    
    def resetThroughputs(self, node: AppNode):
        ''' Reset self.throughput attribute for node and all its children
        '''
        node.throughput = None
        node.incomingThroughput = None
        for child in node.children:
            self.resetThroughputs(node=child)
        return
    
    def addincomingThroughput(self, node: AppNode, incomingThroughput: float):
        print(f'Adding incomingThroughput {incomingThroughput} for task {node.task}')
        if node.incomingThroughput is None:
            node.incomingThroughput = incomingThroughput
        else:
            node.incomingThroughput += incomingThroughput
        return
    
    def addThroughput(self, node: AppNode, throughput: float):
        if node.throughput is None:
            node.throughput = throughput
        else:
            node.throughput += throughput
        return
    
    def getServiceTimeForAllocation(self, allocationPlan: dict,
                                    executionProfiles: pd.DataFrame) -> int:
        ''' Given a proposed allocation plan and execution profiles,
            calculate the service time of the longest path

            An allocation plan is a dict with the following definiton:
            key:    (model variant, batch size, hardware)
            value:  replicas

            NOTE: This function assumes 0 queuing latency
                  InferLine defines service time as "the sum of the processing
                  latencies of all the models on the longest path through the
                  pipeline DAG"
        '''
        # This is what we need to do
        # 1. We first create a mirror graph of the app with each node/task
        #    representing the longest executing instance of that task
        # 2. Then we trace through the graph and find the longest path on it
        #    to find the service time of the allocation plan

        for key in allocationPlan:
            (modelVariant, batchSize, hardware) = key

            row = executionProfiles.loc[(executionProfiles['Model'].str.contains(modelVariant)) &
                                        (executionProfiles['batchsize'] == batchSize) &
                                        (executionProfiles['Accel'] == hardware)]
            
            print(f'modelVariant: {modelVariant}, batchSize: {batchSize}, '
                  f'hardware: {hardware}')

            if len(row) > 0:
                latencyInUSec = int(row['90th_pct'].values[0] * USecInSec)
                print(f'latencyInUSec: {latencyInUSec}')
            else:
                # TODO: what to do if profiled data is not found?
                latencyInUSec = 0
                logging.error('profiled data not found')
        
            task = self.findTaskFromModelVariant(modelVariant=modelVariant)
            node = self.findNodeByTask(task=task)
            self.setMaxLatency(node=node, latency=latencyInUSec)

        maxLatency = self.findLongestPathLatency(node=self.root)

        self.resetMaxLatencies(node=self.root)
        return maxLatency
    
    def findLongestPathLatency(self, node: AppNode) -> int:
        if node.maxLatency is None:
            # TODO: what to do if latency for a task is not set?
            logging.warning(f'findLongestPathLatency(): No maxLatency set for '
                            f'task: {node.task}')
            node.maxLatency = 0

        maxChildLatency = 0

        for child in node.children:
            childLatency = self.findLongestPathLatency(node=child)
            if childLatency > maxChildLatency:
                maxChildLatency = childLatency
        
        return node.maxLatency + maxChildLatency
    
    def resetMaxLatencies(self, node: AppNode):
        ''' Reset self.maxLatency attribute for node and all its children
        '''
        node.maxLatency = None
        for child in node.children:
            self.resetMaxLatencies(child)
        return
    
    def setMaxLatency(self, node: AppNode, latency: int):
        if node.maxLatency is None:
            node.maxLatency = latency
        else:
            if latency > node.maxLatency:
                node.maxLatency = latency
        return
    
    def getDepth(self) -> int:
        ''' Returns depth of the App graph (not counting sink)
        '''
        return self.getDepthHelper(self.root)
    
    def getDepthHelper(self, node: AppNode) -> int:
        if node.task == 'sink':
            return -1
        
        maxSubtreeDepth = 0
        for child in node.children:
            subtreeDepth = self.getDepthHelper(child)
            if subtreeDepth > maxSubtreeDepth:
                maxSubtreeDepth = subtreeDepth
        return maxSubtreeDepth + 1
        
    def print(self):
        ''' Print the entire App graph, starting from the root
        '''
        root = self.root
        print(self.getNodeString(root))
        return
    

def findNodeByTaskHelper(node: AppNode, task: str) -> AppNode:
    if node.task == task:
        return node
    else:
        for child in node.children:
            foundInChild = findNodeByTaskHelper(child, task)
            if foundInChild is not None:
                return foundInChild
        return None
    

def constructSubGraphTopDown(rootNode: AppNode, nodeJson):
    task = nodeJson['task']
    if 'label' in nodeJson:
        label = nodeJson['label']
    else:
        label = None

    node = findNodeByTaskHelper(rootNode, task)
    if node is None:
        node = AppNode(task=task, label=label, modelVariants=nodeJson['model_variants'],
                       children=[])
        
    for childJson in nodeJson['children']:
        childTask = childJson['task']
        if 'label' in childJson:
            childLabel = childJson['label']
        else:
            childLabel = None

        childNode = findNodeByTaskHelper(rootNode, childTask)
        print(f'childTask: {childTask}, childLabel: {childLabel}, childNode: {childNode}')
        if childNode is None:
            childNode = AppNode(task=childTask, label=childLabel,
                                modelVariants=childJson['model_variants'],
                                children=[])

        node.addChild(child=childNode)
    
    # It is important to iteratively construct child nodes and then call this
    # function on children to make sure we build top-down. If we call it
    # recursively and then set children, we are building bottom-up
    for childJson in nodeJson['children']:
        constructSubGraphTopDown(rootNode, childJson)
    return


def constructSubGraphBottomUp(rootNode, nodeJson):
    childNodes = []

    print(f'node: {nodeJson}')
    children = nodeJson['children']
    for child in children:
        childNode = constructSubGraphBottomUp(child)
        childNodes.append(childNode)

    if 'label' in nodeJson:
        label = nodeJson['label']
    else:
        label = None
    
    node = AppNode(task=nodeJson['task'], label=label,
                   modelVariants=nodeJson['model_variants'],
                   children=childNodes)
    return node


def setParentsForSubgraph(node: AppNode):
    ''' For each node in the subgraph, set its parent pointer
    '''
    for child in node.children:
        child.parents.append(node)
        setParentsForSubgraph(child)
    return

    
def registerApplication(filename: str):
    rf = open(filename, mode='r')
    obj = json.load(rf)
    appID = obj['appID']
    appName = obj['appName']
    rootJson = obj['root']
    latencySLOInMSec = obj['latencySLOInMSec']
    print(f'root node: {rootJson}')

    # root = constructSubGraphBottomUp(rootJson)
    # setParentsForSubgraph(root)
    if 'label' in rootJson:
        rootLabel = rootJson['label']
    else:
        rootLabel = None
    rootNode = AppNode(task=rootJson['task'], label=rootLabel,
                       modelVariants=rootJson['model_variants'], children=[])
    
    constructSubGraphTopDown(rootNode=rootNode, nodeJson=rootJson)
    setParentsForSubgraph(rootNode)

    app = App(root=rootNode, appID=appID, appName=appName,
              latencySLOInMSec=latencySLOInMSec)

    print(f'Constructed app graph:')
    app.print()

    return app
