class Task:
    def __init__(self, id, label, name, duration, predecessors, successors, resources):
        self.id = id # a unique integer ID for the task
        self.label = label # a unique label for the task
        self.name = name # the name of the task
        self.duration = duration # time required to complete the task
        self.predecessors = predecessors # maps predecessor IDs to time that
        # must be added to predecessors' starting (-) / end (+) time before current
        # task can start; e.g.: {2: -3, 5: 1} means that current task can only start
        # three time units after task 2 has started and 1 unit after task 5 has ended
        self.successors = successors # maps successor IDs to time after current task can start
        self.resources = resources # maps resource IDs to number of units required
        self.start_time = 0
        self.finish_time = 0

    def __repr__(self):
        return f"Task {self.label}: Name {self.name}, Duration {self.duration}, Start time {self.start_time}, End time {self.finish_time}"

class Resource:
    def __init__(self, id, label, name, type, units):
        self.id = id # a unique integer ID for the resource
        self.label = label # a unique label for the resource
        self.name = name # name of the resource
        self.type = type # whether the resource is fungible or non-fungible
        self.units = units # number of units available for the resource
        self.assigned_tasks = {} # id of tasks using this resource and how many units

    def __repr__(self):
        return f"Resource {self.label}: Name {self.name}, Type {self.type}, Units {self.units}"

class Solution:
    lastID = -1
    def __init__(self):
        Solution.lastID += 1
        self.ID = Solution.lastID
        self.tasks = []
        #self.cost = 0
        self.time = 0

class Test:
    def __init__(self, instanceName, startDate, deadline, dailyPenalty):
        self.instanceName = instanceName
        self.startDate = startDate
        self.deadline = deadline
        self.dailyPenalty = dailyPenalty
        


class Inputs:
    def __init__(self, name, nTasks, nResources, tasks, resources):
        self.name = name
        self.nTasks = nTasks
        self.nResources = nResources
        self.tasks = tasks
        self.resources = resources