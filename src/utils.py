import pandas as pd
import sys, os
from classes import Task, Resource, Inputs

# Get the directory of the currently executing script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Adjust the OUTPUTS_DIR to be an absolute path
OUTPUTS_DIR = os.path.join(current_directory, '../outputs')
INPUTS_DIR = os.path.join(current_directory, '../inputs')
TESTS_DIR = os.path.join(current_directory, '../tests')


     
def readTests(fileName):
    tests = []
    path_to_file = os.path.join(TESTS_DIR, fileName)
 
    with open(path_to_file) as f:
        for line in f:
            tokens = line.split("\t")
            if tokens and '#' not in tokens[0]:
                instance_name = tokens[0].strip()
                start_date = tokens[1].strip() if len(tokens) > 1 else None
                deadline = tokens[2].strip() if len(tokens) > 2 else None
                daily_penalty = tokens[3].strip() if len(tokens) > 3 else None
                test = Test(instance_name, start_date, deadline, daily_penalty)
                tests.append(test)
    return tests


def readInputs(instanceName):

    path_to_file = os.path.join(INPUTS_DIR, instanceName + ".xlsx")
    
    # Read tasks and resources into DataFrames using the adjusted path
    tasks_df = pd.read_excel(path_to_file, sheet_name="Tasks", dtype={"ID" : str, "Predecessors" : str, "Successors" : str}, na_filter=False)
    resources_df = pd.read_excel(path_to_file, sheet_name="Resources")
    
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
                    pred_id = tasks_df.loc[tasks_df["ID"] == label].index[0]
                    predecessors[pred_id] = int(pred[fc_index + 3:])
                elif cc_index != -1:
                    label = pred[:cc_index]
                    pred_id = tasks_df.loc[tasks_df["ID"] == label].index[0]
                    predecessors[pred_id] = -int(pred[cc_index + 3:])
                else:
                    pred_id = tasks_df.loc[tasks_df["ID"] == pred].index[0]
                    predecessors[pred_id] = 0

        for succ in row['Successors'].split(";"):
            if succ:
                fc_index = succ.find("FC")
                cc_index = succ.find("CC")
                if fc_index != -1:
                    label = succ[:fc_index]
                    succ_id = tasks_df.loc[tasks_df["ID"] == label].index[0]
                    successors[succ_id] = int(succ[fc_index + 3:])
                elif cc_index != -1:
                    label = succ[:cc_index]
                    succ_id = tasks_df.loc[tasks_df["ID"] == label].index[0]
                    successors[succ_id] = -int(succ[cc_index + 3:])
                else:
                    succ_id = tasks_df.loc[tasks_df["ID"] == succ].index[0]
                    successors[succ_id] = 0

        for j, res in enumerate(row[6 : len(tasks_df.columns)]):
            if res:
                resources[j] = float(res)

        task = Task(i, row['ID'], row['Name'], row['Duration'], predecessors, successors, resources)
        tasks.append(task)


    # Create list of resource objects
    resources = []
    for i, row in resources_df.iterrows():
        resource = Resource(i, row['ID'], row['Name'], row['Type'], row['Units'])
        resources.append(resource)

    inputs = Inputs(instanceName, len(tasks), len(resources), tasks, resources)
    return inputs


def get_predecessor_notation(task_label, extra_time):
    if extra_time == 0:
        return task_label
    elif extra_time > 0:
        return f"{task_label}FC+{extra_time}"
    else:
        return f"{task_label}FC-{extra_time}"

def printSolutionToExcel(solution, project_name):
    # Create a DataFrame to store the solution tasks
    data = {
        "Task Label": [task.label for task in solution.tasks],
        "Task Name": [task.name for task in solution.tasks],
        "Duration": [task.duration for task in solution.tasks],
        "Start Time": [task.start_time for task in solution.tasks],
        "Finish Time": [task.finish_time for task in solution.tasks],
    }

    # Collect predecessors and successors data
    predecessors_data = []
    successors_data = []
    for task in solution.tasks:
        predecessors_data.append(", ".join(get_predecessor_notation(inputs.tasks[pred].label, extra_time) for pred, extra_time in task.predecessors.items()))
        successors_data.append(", ".join(get_predecessor_notation(inputs.tasks[succ].label, extra_time) for succ, extra_time in task.successors.items()))

    # Add predecessors and successors data to the DataFrame
    data["Predecessors"] = predecessors_data
    data["Successors"] = successors_data

    df = pd.DataFrame(data)

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Define the output file path in the output directory with the project name as a prefix
    output_file_path = os.path.join(OUTPUTS_DIR, f"{project_name}_output_solution.xlsx")


    # Write the DataFrame to an Excel file in the output directory
    df.to_excel(output_file_path, sheet_name="Solution", index=False)

    # Append cost and time to the Excel file as additional information
    with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:
        additional_info_df = pd.DataFrame({"Time": [solution.time]})
        additional_info_df.to_excel(writer, sheet_name="Additional Info", index=False)


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
                successor = inputs.tasks[successor_id]
                queue.append(successor)
    return sorted_tasks


"""
Topological Ordering and Resource Allocation (TORA) heuristic.
"""
def TORA_Heuristic(inputs):
    solution = Solution()
    
    # Initialize resources availability
    resources_availability = {resource.id: resource.units for resource in inputs.resources}

    # Initialize dictionary to keep track of resources used by each task
    resources_used = {task.id: set() for task in inputs.tasks}

    # Check if any task demands more units of a resource than its total capacity
    for task in inputs.tasks:
        for resource_id, units in task.resources.items():
            if units > resources_availability[resource_id]:
                sys.exit(f"[ERROR]: Task {task.label} demands {units} units of resource {resource_id}, but only {resources_availability[resource_id]} units are available.")

    # Topologically sort the tasks
    sorted_tasks = topological_sort(inputs.tasks)

    # Loop over the tasks in topological order
    for task in sorted_tasks:
        # Calculate earliest start time for task considering predecessor dependencies
        earliest_start_time = 0
        for pred_id, extra_time in task.predecessors.items():
            pred_task = inputs.tasks[pred_id]
            if extra_time >= 0:
                # Add extra time to the ending time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.finish_time + extra_time)
            else:
                # Add extra time to the starting time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.start_time + abs(extra_time))

        # Refine earliest start time for task considering resource availability
        for resource_id, units in task.resources.items():
            while resources_availability[resource_id] < units:
                resource = inputs.resources[resource_id]
                # Sort assigned tasks to resource by finish time
                assigned_tasks = [inputs.tasks[task_id] for task_id, _ in resource.assigned_tasks.items()]
                assigned_tasks.sort(key=lambda t: t.finish_time)
                if not assigned_tasks:
                    raise ValueError(f"No more assigned tasks available for resource {resource_id}")
                assigned_task = assigned_tasks.pop(0)

                # Release resources used by task
                for rel_resource_id in resources_used[assigned_task.id]:
                    rel_resource = inputs.resources[rel_resource_id]
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
            resource = inputs.resources[resource_id]
            resource.assigned_tasks[task.id] = units
            resources_used[task.id].add(resource_id)

        # Add task to project schedule
        solution.tasks.append(task)

    solution.time = task.finish_time
    return solution


"""
Network diagram generator
"""
def network_diagram(inputs):
    solution = Solution()
    
    # Topologically sort the tasks. A topological sort is an algorithm that takes a directed
    # graph and returns a linear ordering of its vertices (nodes) such that, for every
    # directed edge (u, v) from vertex u to vertex v, u comes before v in the ordering
    sorted_tasks = topological_sort(inputs.tasks)

    # Loop over the tasks in topological order
    for task in sorted_tasks:
        # Calculate earliest start time for task considering predecessor dependencies
        earliest_start_time = 0
        for pred_id, extra_time in task.predecessors.items():
            pred_task = inputs.tasks[pred_id]
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
