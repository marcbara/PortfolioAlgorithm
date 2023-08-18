import pandas as pd
import sys, os
import configparser
from classes import Task, Resource, Project, Solution, Portfolio
from datetime import datetime, timedelta

# Get the directory of the currently executing script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to config.ini
class CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr
config_file_path = os.path.join(current_directory, 'config.ini')
config = CaseSensitiveConfigParser()
read_files = config.read(config_file_path)
if not read_files:
    print("Failed to read the config.ini file.")


# Adjust directory paths to be absolute paths relative to the current_directory
OUTPUTS_DIR = os.path.join(current_directory, config['PATHS']['OUTPUTS_DIR'])
INPUTS_DIR = os.path.join(current_directory, config['PATHS']['INPUTS_DIR'])

# Constants
RESOURCES_SHEET_NAME = "Resources"
PORTFOLIO_FILE = config['PATHS']['PORTFOLIO_FILE']

def get_earliest_date(dates):
    date_format = "%d-%m-%Y"
    parsed_dates = [datetime.strptime(date, date_format) for date in dates]
    return min(parsed_dates)

def get_labor_days_difference(start_date, end_date):
    current_date = start_date
    labor_days = 0
    while current_date < end_date:
        if current_date.weekday() < 5:  # 0-4 denotes Monday to Friday
            labor_days += 1
        current_date += timedelta(days=1)
    return labor_days

def adjust_task_dates_by_offset(task: Task, start_offset=0):
    """
    Adjust the start_time and finish_time attributes of a task based on the provided start offset.
    """
    task.start_time += start_offset
    task.finish_time += start_offset

def set_task_absolute_dates(task: Task, portfolio_start_date: str):
    date_format = "%d-%m-%Y"
    portfolio_date = datetime.strptime(portfolio_start_date, date_format)
    task.start_date = (portfolio_date + timedelta(days=task.start_time)).strftime(date_format)
    task.finish_date = (portfolio_date + timedelta(days=task.finish_time)).strftime(date_format)

def set_project_task_dates(project: Project, portfolio_start_date: str):
    for task in project.tasks:
        set_task_absolute_dates(task, portfolio_start_date)


def readProjects():
    project_data = {}
    for project_name, project_values in config['PROJECTS'].items():
        start_date, deadline, daily_penalty = project_values.split(',')
        project_data[project_name] = datetime.strptime(start_date, "%d-%m-%Y")

    # Find the earliest start date
    earliest_start = min(project_data.values())

    projects_list = []
    for project_name, start_date in project_data.items():
        _, deadline, daily_penalty = config['PROJECTS'][project_name].split(',')
        start_offset = get_labor_days_difference(earliest_start, start_date)
        project = Project(project_name, start_date.strftime("%d-%m-%Y"), deadline, float(daily_penalty), start_offset)
        projects_list.append(project)
    
    # Create the Portfolio object
    portfolio_start_date = earliest_start.strftime("%d-%m-%Y")
    portfolio = Portfolio(portfolio_start_date)
    for project in projects_list:
        portfolio.add_project(project)

    return portfolio


def readInputs(project):

    instanceName = project.instanceName

    # Reading resources from the RESOURCES_SHEET_NAME
    resources_df = pd.read_excel(os.path.join(INPUTS_DIR, PORTFOLIO_FILE), sheet_name=RESOURCES_SHEET_NAME)

    # Reading tasks from a sheet specific to the project (based on the instanceName)
    tasks_df = pd.read_excel(os.path.join(INPUTS_DIR, PORTFOLIO_FILE), sheet_name=instanceName, dtype={"ID": str, "Predecessors": str, "Successors": str}, na_filter=False)

    # Create a dictionary to map task labels to their index
    task_label_to_index = {label: idx for idx, label in enumerate(tasks_df["ID"])}
    
    # Create a dictionary to map resource labels to their index
    resource_label_to_index = {row['ID']: i for i, row in resources_df.iterrows()}

    # Create list of task objects
    tasks = []
    for i, row in tasks_df.iterrows():
        predecessors = {}
        successors = {}
        resources = {}

        for pred in row['Predecessors'].split(";"):
            if pred:
                fc_index = pred.find("FC")
                cc_index = pred.find("CC")
                if fc_index != -1:
                    label = pred[:fc_index]
                    pred_id = task_label_to_index[label]
                    predecessors[pred_id] = int(pred[fc_index + 3:])
                elif cc_index != -1:
                    label = pred[:cc_index]
                    pred_id = task_label_to_index[label]
                    predecessors[pred_id] = -int(pred[cc_index + 3:])
                else:
                    pred_id = task_label_to_index[pred]
                    predecessors[pred_id] = 0

        for succ in row['Successors'].split(";"):
            if succ:
                fc_index = succ.find("FC")
                cc_index = succ.find("CC")
                if fc_index != -1:
                    label = succ[:fc_index]
                    succ_id = task_label_to_index[label]
                    successors[succ_id] = int(succ[fc_index + 3:])
                elif cc_index != -1:
                    label = succ[:cc_index]
                    succ_id = task_label_to_index[label]
                    successors[succ_id] = -int(succ[cc_index + 3:])
                else:
                    succ_id = task_label_to_index[succ]
                    successors[succ_id] = 0
        
        # Adjusting how resources are assigned to tasks
        for j, (col_name, res) in enumerate(row.items()):
            if j >= 6 and res:
                try:
                    resource_id = resource_label_to_index[col_name]
                    resources[resource_id] = float(res)
                except ValueError:
                    raise ValueError(f"Invalid resource value '{res}' for task {row['ID']} in resource column {col_name}. Expected a numeric value.")

        task = Task(i, row['ID'], row['Name'], row['Duration'], predecessors, successors, resources, project)
        adjust_task_dates_by_offset(task, project.start_offset)
        tasks.append(task)

    # Create list of resource objects
    resources_list = []
    for i, row in resources_df.iterrows():
        resource = Resource(i, row['ID'], row['Name'], row['Type'], row['Units'])
        resources_list.append(resource)

    # Check if any task demands more units of a resource than its total capacity
    resources_availability = {resource.id: resource.units for resource in resources_list}
    for task in tasks:
        for resource_id, units in task.resources.items():
            if units > resources_availability[resource_id]:
                resource_label = resources_list[resource_id].name   # Fetching the name attribute from the Resource object
                raise ValueError(f"Project {instanceName} - Task {task.label} {task.name} demands {units} units of resource {resource_label}, but only {resources_availability[resource_id]} units are available.")

    # Append tasks and resources to the given project.
    project.tasks.extend(tasks)
    project.resources.extend(resources_list)

    return project

def ProjectToDF(project):
    # Create a DataFrame to store the project tasks
    data = {
        "Task Label": [task.label for task in project.tasks],
        "Task Name": [task.name for task in project.tasks],
        "Duration": [task.duration for task in project.tasks],
        "Start Time": [task.start_time for task in project.tasks],
        "Finish Time": [task.finish_time for task in project.tasks],
        "Start Date": [task.start_date for task in project.tasks],
        "Finish Date": [task.finish_date for task in project.tasks],
    }

    # Collect predecessors and successors data
    predecessors_data = []
    successors_data = []
    for task in project.tasks:
        predecessors_data.append(", ".join(get_predecessor_notation(project.tasks[pred].label, extra_time) for pred, extra_time in task.predecessors.items()))
        successors_data.append(", ".join(get_predecessor_notation(project.tasks[succ].label, extra_time) for succ, extra_time in task.successors.items()))

    # Add predecessors and successors data to the DataFrame
    data["Predecessors"] = predecessors_data
    data["Successors"] = successors_data

    df = pd.DataFrame(data)
    return df



def write_solutions_to_excel(dfs, sheet_names):
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Write all dataframes to a single Excel file with different sheets
    output_filename = f"{PORTFOLIO_FILE.split('.')[0]}_solutions.xlsx"
    with pd.ExcelWriter(os.path.join(OUTPUTS_DIR, output_filename)) as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def get_predecessor_notation(task_label, extra_time):
    if extra_time == 0:
        return task_label
    elif extra_time > 0:
        return f"{task_label}FC+{extra_time}"
    else:
        return f"{task_label}FC-{extra_time}"


def topological_sort(tasks):
    # Create a dictionary to store the number of incoming edges for each task
    in_degree = {task.id: len(task.predecessors) for task in tasks}
    # Initialize the queue with the tasks that have no incoming edges
    queue = [task for task in tasks if in_degree[task.id] == 0]
    # Perform topological sorting
    sorted_tasks = []
    # while queue is not empty
    while queue:
        # Get the next task in the queue and add it to the sorted list
        task = queue.pop(0)
        sorted_tasks.append(task)
        # Decrement the incoming edge count for each successor of task
        for successor_id, _ in task.successors.items():
            in_degree[successor_id] -= 1
            # If the successor has no more incoming edges, add it to the queue
            if in_degree[successor_id] == 0:
                successor = tasks[successor_id]
                queue.append(successor)
    return sorted_tasks


"""
Topological Ordering and Resource Allocation (TORA) heuristic.
"""
def TORA_Heuristic(project):
    solution = Solution()
    
    # Initialize resources availability
    resources_availability = {resource.id: resource.units for resource in project.resources}

    # Initialize dictionary to keep track of resources used by each task
    resources_used = {task.id: set() for task in project.tasks}

    # Check if any task demands more units of a resource than its total capacity
    for task in project.tasks:
        for resource_id, units in task.resources.items():
            if units > resources_availability[resource_id]:
                sys.exit(f"[ERROR]: Task {task.label} demands {units} units of resource {resource_id}, but only {resources_availability[resource_id]} units are available.")

    # Topologically sort the tasks
    sorted_tasks = topological_sort(project.tasks)

    # Loop over the tasks in topological order
    for task in sorted_tasks:
        # Calculate earliest start time for task considering predecessor dependencies
        earliest_start_time = task.start_time # Use the task's current start time as the base
        for pred_id, extra_time in task.predecessors.items():
            pred_task = project.tasks[pred_id]
            if extra_time >= 0:
                # Add extra time to the ending time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.finish_time + extra_time)
            else:
                # Add extra time to the starting time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.start_time + abs(extra_time))

        # Refine earliest start time for task considering resource availability
        for resource_id, units in task.resources.items():
            while resources_availability[resource_id] < units:
                resource = project.resources[resource_id]
                # Sort assigned tasks to resource by finish time
                assigned_tasks = [project.tasks[task_id] for task_id, _ in resource.assigned_tasks.items()]
                assigned_tasks.sort(key=lambda t: t.finish_time)
                if not assigned_tasks:
                    raise ValueError(f"No more assigned tasks available for resource {resource_id}")
                assigned_task = assigned_tasks.pop(0)

                # Release resources used by task
                for rel_resource_id in resources_used[assigned_task.id]:
                    rel_resource = project.resources[rel_resource_id]
                    # Update resources assigned tasks
                    rel_units = rel_resource.assigned_tasks[assigned_task.id]
                    resources_availability[rel_resource_id] += rel_units
                    del rel_resource.assigned_tasks[assigned_task.id]

                # Reset resources used by task
                resources_used[assigned_task.id] = set()

                # Update earliest start time considering task ending time
                earliest_start_time = max(earliest_start_time, assigned_task.finish_time)

        # Update finish time of task
        task.start_time = earliest_start_time
        task.finish_time = earliest_start_time + task.duration

        # Assign resources to task
        for resource_id, units in task.resources.items():
            resources_availability[resource_id] -= units
            resource = project.resources[resource_id]
            resource.assigned_tasks[task.id] = units
            resources_used[task.id].add(resource_id)

        # Add task to project schedule
        solution.tasks.append(task)

    solution.time = task.finish_time
    return solution


"""
Network diagram generator
"""
def network_diagram(project):
    solution = Solution()
    
    # Topologically sort the tasks. A topological sort is an algorithm that takes a directed
    # graph and returns a linear ordering of its vertices (nodes) such that, for every
    # directed edge (u, v) from vertex u to vertex v, u comes before v in the ordering
    sorted_tasks = topological_sort(project.tasks)

    # Loop over the tasks in topological order
    for task in sorted_tasks:
        # Calculate earliest start time for task considering predecessor dependencies
        earliest_start_time = task.start_time # Use the task's current start time as the base
        for pred_id, extra_time in task.predecessors.items():
            pred_task = project.tasks[pred_id]
            if extra_time >= 0:
                # Add extra time to the ending time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.finish_time + extra_time)
            else:
                # Add extra time to the starting time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.start_time + abs(extra_time))
        
        # Update finish time of task
        task.start_time = earliest_start_time
        task.finish_time = earliest_start_time + task.duration

        # Add task to project schedule
        solution.tasks.append(task)
 
    solution.time = task.finish_time
    return solution
