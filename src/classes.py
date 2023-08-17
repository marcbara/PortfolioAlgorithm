class Project:
    """
    Represents a project instance with its associated properties.
    
    Attributes:
        instanceName (str): Name of the project instance.
        startDate (str): Start date for the project.
        deadline (str): Deadline for the project.
        dailyPenalty (str): Daily penalty for exceeding the deadline.
    """
    def __init__(self, instanceName, startDate, deadline, dailyPenalty):
        self.instanceName = instanceName
        self.startDate = startDate
        self.deadline = deadline
        self.dailyPenalty = dailyPenalty

    def __repr__(self):
        return (f"Project({self.instanceName!r}, Start: {self.startDate}, "
                f"Deadline: {self.deadline}, Daily Penalty: {self.dailyPenalty})\n")


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
        project (Project): The project to which this task belongs.
    """
    def __init__(self, id, label, name, duration, predecessors, successors, resources, project=None):
        self.id = id
        self.label = label
        self.name = name
        self.duration = duration
        self.predecessors = predecessors
        self.successors = successors
        self.resources = resources
        self.start_time = 0
        self.finish_time = 0
        self.project = project

    def __repr__(self):
        predecessor_labels = ", ".join(str(label) for label in self.predecessors.keys())
        return (f"Task(ID: {self.id}, Label: {self.label}, Name: {self.name}, "
                f"Duration: {self.duration}, Start time: {self.start_time}, "
                f"End time: {self.finish_time}, Project: {self.project.instanceName if self.project else 'None'}, "
                f"Predecessors IDs: [{predecessor_labels}])\n")


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
        self.time = 0

  
class Inputs:
    """
    Represents the inputs for a given instance.
    
    Attributes:
        name (str): Name of the Project instance.
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