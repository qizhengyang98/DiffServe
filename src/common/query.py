class Query:
    def __init__(self, requestID, queryID, userID, applicationID, taskID, prompt, data, resultQualified, 
                 startTimestamp, queuedTimeStamp, latencyBudget,
                 sequenceNum):
        self.requestID = requestID
        self.queryID = queryID
        self.userID = userID
        self.applicationID = applicationID
        self.taskID = taskID
        self.prompt = prompt
        self.data = data
        self.resultQualified = resultQualified
        self.startTimestamp = startTimestamp
        self.queuedTimestamp = queuedTimeStamp
        self.latencyBudget = latencyBudget
        self.sequenceNum = sequenceNum
        return

class QueryResults:
    def __init__(self, queryID, prompt, result, result_qualified):
        self.queryID = queryID
        self.prompt = prompt
        self.result = result
        self.result_qualified = result_qualified
        