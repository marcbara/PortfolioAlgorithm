class Task:
    """
    Represents a task in a project with its associated properties.
    
    Attributes:
        id (int): A unique identifier for the task.
        label (str): A unique label for the task.
        name (str): Name or title of the task.
        duration (int): Time required to complete the task.
        predecessors (dict): Maps predecessor task IDs to time offsets.
        successors (dict): Maps successor task IDs to time offsets.
        resources (dict): Maps resource IDs to required number of units.
        start_time (int): Calculated start time of the task.
        finish_time (int): Calculated finish time of the task.
    """
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
    """
    Represents a resource with its associated properties and assignments.
    
    Attributes:
        id (int): A unique identifier for the resource.
        label (str): A unique label for the resource.
        name (str): Name of the resource.
        type (str): Type of the resource (fungible or non-fungible).
        units (int): Total available units of the resource.
        assigned_tasks (dict): Maps task IDs to the number of units assigned.
    """
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
    """
    Represents a potential solution or plan for a set of tasks.
    
    Attributes:
        ID (int): A unique identifier for the solution.
        tasks (list): A list of task objects in the solution.
        time (int): Total time for the solution.
    """
    lastID = -1
    def __init__(self):
        Solution.lastID += 1
        self.ID = Solution.lastID
        self.tasks = []
        #self.cost = 0
        self.time = 0

class Test:
    """
    Represents a test instance with its associated properties.
    
    Attributes:
        instanceName (str): Name of the test instance.
        startDate (str): Start date for the test.
        deadline (str): Deadline for the test.
        dailyPenalty (str): Daily penalty for exceeding the deadline.
    """
    def __init__(self, instanceName, startDate, deadline, dailyPenalty):
        self.instanceName = instanceName
        self.startDate = startDate
        self.deadline = deadline
        self.dailyPenalty = dailyPenalty
        


class Inputs:
    """
    Represents the inputs for a given instance.
    
    Attributes:
        name (str): Name of the instance.
        nTasks (int): Number of tasks in the instance.
        nResources (int): Number of resources in the instance.
        tasks (list): A list of task objects for the instance.
        resources (list): A list of resource objects for the instance.
    """
    def __init__(self, name, nTasks, nResources, tasks, resources):
        self.name = name
        self.nTasks = nTasks
        self.nResources = nResources
        self.tasks = tasks
        self.resources = resources