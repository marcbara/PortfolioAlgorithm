from datetime import datetime


class Portfolio:
    """
    Represents a collection of project instances.
    
    Attributes:
        projects (list): List of projects in the portfolio.
        start_date (str): Absolute start date for the portfolio.
    """
    def __init__(self, start_date):
        self.projects = []
        self.start_date = start_date

    def add_project(self, project):
        """
        Adds a project to the portfolio.
        
        Parameters:
            project (Project): The project instance to be added.
        """
        self.projects.append(project)

    def __repr__(self):
        return f"Portfolio(Start Date: {self.start_date}, Projects: {[project.instanceName for project in self.projects]})"


class Project:
    """
    Represents a project instance with its associated properties, tasks, and resources.
    
    Attributes:
        instanceName (str): Name of the project instance.
        startDate (str): Start date for the project.
        deadline (str): Deadline for the project.
        dailyPenalty (str): Daily penalty for exceeding the deadline.
        start_offset (int): Number of labor days from the earliest project start date.
        tasks (list): List of tasks associated with the project.
        resources (list): List of resources available for the project.
    """
    def __init__(self, instanceName, startDate, deadline, dailyPenalty, start_offset=0, dailyWaterConsumption=0):
        self.instanceName = instanceName
        self.startDate = startDate
        self.deadline = deadline
        self.dailyPenalty = dailyPenalty
        self.start_offset = start_offset
        self.tasks = []
        self.resources = []
        self.dailyWaterConsumption = dailyWaterConsumption
        
    def get_delivery_date(self):
        """Compute the delivery date of the project based on the finish dates of its tasks."""
        latest_finish_date = max(datetime.strptime(task.finish_date, '%d-%m-%Y').date() for task in self.tasks)
        latest_finish_date_string = latest_finish_date.strftime('%d-%m-%Y')
        return latest_finish_date_string

    def compute_penalty(self):
        from utils import get_labor_days_difference
        """Compute the penalty for the actual delivery date of the project."""
        # Get the actual delivery date from the tasks' finish dates
        delivery_date = self.get_delivery_date()
        
        # Convert the delivery date and the deadline to Python date objects
        delivery_date_obj = datetime.strptime(delivery_date, '%d-%m-%Y').date()
        deadline_date_obj = datetime.strptime(self.deadline, '%d-%m-%Y').date()
        
        # Compute the difference in labor days between the delivery date and the deadline
        days_late = get_labor_days_difference(deadline_date_obj, delivery_date_obj)
        
        # If the project is delivered on or before the deadline, the penalty is zero
        if days_late <= 0:
            return 0
        
        # If the project is delivered after the deadline, compute the total penalty
        return days_late * self.dailyPenalty
    
    def __repr__(self):
        return (f"Project({self.instanceName!r}, Start: {self.startDate}, "
                f"Deadline: {self.deadline}, Daily Penalty: {self.dailyPenalty}, "
                f"Start Offset: {self.start_offset} labor days, "
                f"Daily Water Consumption: {self.dailyWaterConsumption} m3\n")


class Task:
    """
    Represents a task in a project with its associated properties.
    
    Attributes:
        id (int): A unique identifier for the task.
        label (str): A unique label for the task.
        name (str): Name or title of the task.
        duration (int): Time required to complete the task.
        predecessors (dict): Maps predecessor task IDs to time offsets.
        external_predecessors (dict): Maps external predecessor task IDs to time offsets.
        successors (dict): Maps successor task IDs to time offsets.
        resources (dict): Maps resource IDs to required number of units.
        start_time (int): Start time of the task, in days, wrt portfolio start
        finish_time (int): Finish time of the task, in days, wrt portfolio start.
        project (Project): The project to which this task belongs.
        start_date (str): Absolute start calendar date
        finish_date (str): Absolute finish calendar date
        section (str): The section of the project to which this task belongs.
    """
    def __init__(self, id, label, name, duration, predecessors, external_predecessors, successors, resources, project=None, section=None):
        self.id = id
        self.label = label
        self.name = name
        self.duration = duration
        self.predecessors = predecessors
        self.external_predecessors = external_predecessors
        self.successors = successors
        self.resources = resources
        self.start_time = 0
        self.finish_time = 0
        self.project = project
        self.start_date = None
        self.finish_date = None
        self.section = section

    def __repr__(self):
        predecessor_labels = ", ".join(str(label) for label in self.predecessors.keys())
        external_predecessor_labels = ", ".join(str(label) for label in self.external_predecessors.keys())
        successors_labels = ", ".join(str(label) for label in self.successors.keys())
        return (f"Task(ID: {self.id}, Label: {self.label}, Name: {self.name}, "
                f"Duration: {self.duration}, Start time: {self.start_time}, "
                f"End time: {self.finish_time}, Project: {self.project.instanceName if self.project else 'None'}, "
                f"Start date: {self.start_date}, Finish date: {self.finish_date}, "
                f"Predecessors IDs: [{predecessor_labels}], "
                f"External Predecessors IDs: [{external_predecessor_labels}], "
                f"Successors IDs: [{successors_labels}], "
                f"Section: {self.section})\n")


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

    def to_project(self, original_project):
        # Generate tasks from the solution's tasks
        solved_tasks = [Task(task.id, task.label, task.name, task.duration, task.predecessors, task.external_predecessors,
                             task.successors, task.resources, task.project, task.section) 
                        for task in self.tasks]
        
        # Set start and finish times based on the solution
        for task, solved_task in zip(original_project.tasks, solved_tasks):
            solved_task.start_time = task.start_time
            solved_task.finish_time = task.finish_time

        # Create a new project instance with the same attributes as the original
        solved_project = Project(original_project.instanceName, original_project.startDate, 
                                 original_project.deadline, original_project.dailyPenalty, 
                                 original_project.start_offset, original_project.dailyWaterConsumption)
        
        # Assign the new list of tasks to the project
        solved_project.tasks = solved_tasks

        # Copy the resources from the original project to the solved project
        solved_project.resources = original_project.resources.copy()

        return solved_project

