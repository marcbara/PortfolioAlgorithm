import pandas as pd
import sys, os
import configparser
from classes import Task, Resource, Project, Solution, Portfolio
from datetime import datetime, timedelta
import logging
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openai


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

secret_config = CaseSensitiveConfigParser()
read_secret_files = secret_config.read(os.path.join(current_directory, 'secrets.ini'))
if not read_secret_files:
    print("Failed to read the secrets.ini file.")

openai.api_key = secret_config.get("secrets", "openai_api_key")


# Adjust directory paths to be absolute paths relative to the current_directory
OUTPUTS_DIR = os.path.join(current_directory, config['PATHS']['OUTPUTS_DIR'])
INPUTS_DIR = os.path.join(current_directory, config['PATHS']['INPUTS_DIR'])

# Constants
RESOURCES_SHEET_NAME = "Resources"
PORTFOLIO_FILE = config['PATHS']['PORTFOLIO_FILE']

# Settings
generate_gantts = config.getboolean('SETTINGS', 'GenerateGantt', fallback=False)
generate_ai_insights = config.getboolean('SETTINGS', 'GenerateAIInsights', fallback=False)

# Set up logging
portfolio_file_basename = os.path.basename(PORTFOLIO_FILE)
portfolio_name, _ = os.path.splitext(portfolio_file_basename)
log_filename = os.path.join(OUTPUTS_DIR, f"{portfolio_name}_log.txt")
AI_insights_filename = os.path.join(OUTPUTS_DIR, f"{portfolio_name}_AI_insights.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s', filemode='w')


def get_earliest_date(dates):
    """
    Find and return the earliest date from a list of date strings.

    Args:
        dates (list): List of date strings in the format "dd-mm-yyyy".

    Returns:
        datetime: The earliest date found in the list.
    """
    date_format = "%d-%m-%Y"
    parsed_dates = [datetime.strptime(date, date_format) for date in dates]
    return min(parsed_dates)

def get_labor_days_difference(start_date, end_date):
    """
    Calculate the number of labor days between two dates, excluding weekends (Saturday and Sunday).

    Args:
        start_date (datetime.date): The starting date.
        end_date (datetime.date): The ending date.

    Returns:
        int: The number of labor days between the two dates.
    """
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

    Args:
        task (Task): The task to adjust.
        start_offset (int, optional): The offset to apply to the task's start and finish times. Default is 0.
    """
    task.start_time += start_offset
    task.finish_time += start_offset


def set_task_absolute_dates(task: Task, portfolio_start_date: str):
    """
    Set the absolute start and finish dates for a task based on the portfolio's start date.

    Args:
        task (Task): The task to update.
        portfolio_start_date (str): The start date of the portfolio in "dd-mm-yyyy" format.
    """
    date_format = "%d-%m-%Y"
    portfolio_date = datetime.strptime(portfolio_start_date, date_format)

    # Calculate the start date by adding the business day offset
    start_date = portfolio_date + pd.offsets.BDay(task.start_time)

    if task.duration == 0:
        finish_date = start_date
    else:
        # Calculate the finish date by adding the business day offset and duration
        finish_date = start_date + pd.offsets.BDay(task.duration - 1)

    # Set the start and finish dates in the task
    task.start_date = start_date.strftime(date_format)
    task.finish_date = finish_date.strftime(date_format)



def set_project_task_dates(project: Project, portfolio_start_date: str):
    """
    Set the absolute start and finish dates for tasks in a project based on the portfolio's start date.

    Args:
        project (Project): The project whose tasks' dates need to be set.
        portfolio_start_date (str): The start date of the portfolio in "dd-mm-yyyy" format.
    """
    for task in project.tasks:
        set_task_absolute_dates(task, portfolio_start_date)


def read_projects():
    """
    Read project information from the config file and create a Portfolio object.

    Returns:
        Portfolio: The created portfolio containing projects and their associated information.
    """
    projects_list = []
    earliest_start = None

    for project_name, project_values in config['PROJECTS'].items():
        start_date_str, deadline, daily_penalty, daily_water_consumption = project_values.split(',')
        start_date = datetime.strptime(start_date_str, "%d-%m-%Y")
        
        # Update the earliest start date
        if earliest_start is None or start_date < earliest_start:
            earliest_start = start_date
        
        start_offset = get_labor_days_difference(earliest_start, start_date)
        project = Project(project_name, start_date_str, deadline, float(daily_penalty), start_offset, float(daily_water_consumption))
        projects_list.append(project)

    # Create the Portfolio object
    portfolio_start_date = earliest_start.strftime("%d-%m-%Y")
    portfolio = Portfolio(portfolio_start_date)
    for project in projects_list:
        portfolio.add_project(project)

    return portfolio



def read_inputs(project):
    """
    Reads project tasks and resources data from an Excel file and populates the given project instance.
    
    This function expects the Excel file to have specific sheets and columns layout:
    - Resources data should be in a sheet named by the value in `RESOURCES_SHEET_NAME`.
    - Tasks data should be in a sheet named by the project's `instanceName`.

    The function processes tasks' relations, resources assignments, and checks resources' availability.
    Invalid or inconsistent inputs will raise a ValueError.
    
    Args:
        project (Project): The project instance to be populated with tasks and resources.
        
    Returns:
        Project: The populated project instance.
    
    Raises:
        ValueError: If there's an inconsistency in resources' assignment or if any task demands more units 
                   of a resource than its total capacity.
                   
    Notes:
        The Excel file's path is constructed using `INPUTS_DIR` and `PORTFOLIO_FILE` constants.
        The function expects certain column names in the Excel sheets, and any deviations may lead to errors.
    """
    instanceName = project.instanceName

    # Reading data from Excel
    resources_df = pd.read_excel(os.path.join(INPUTS_DIR, PORTFOLIO_FILE), sheet_name=RESOURCES_SHEET_NAME)
    tasks_df = pd.read_excel(os.path.join(INPUTS_DIR, PORTFOLIO_FILE), sheet_name=instanceName, dtype={"ID": str, "Predecessors": str, "Successors": str, "External Predecessors": str, "Section": str}, na_filter=False)

    # Mapping labels to indices
    task_label_to_index = {label: idx for idx, label in enumerate(tasks_df["ID"])}
    resource_label_to_index = {row['ID']: i for i, row in resources_df.iterrows()}

    def process_task_relations(task_string, is_external=False):
        relations = {}
        for task in task_string.split(";"):
            if task:
                fc_index = task.find("FC")
                cc_index = task.find("CC")

                if fc_index != -1:
                    label = task[:fc_index]
                    lag = int(task[fc_index + 3:])
                elif cc_index != -1:
                    label = task[:cc_index]
                    lag = -int(task[cc_index + 3:])
                else:
                    label = task
                    lag = 0

                if is_external:
                    proj_id, label = label.split("-")
                    key = f"{proj_id}-{label}"
                    relations[key] = lag
                else:
                    task_id = task_label_to_index[label]
                    relations[task_id] = lag
        return relations

    tasks = []
    for i, row in tasks_df.iterrows():
        predecessors = process_task_relations(row['Predecessors'])
        successors = process_task_relations(row['Successors'])
        external_predecessors = process_task_relations(row['External Predecessors'], is_external=True)
        
        # Adjusting how resources are assigned to tasks
        resources = {}
        excluded_cols = ["ID", "Section", "Name", "Duration", "Predecessors", "Successors", "External Predecessors"]
        for col_name, res in row.items():
            if col_name not in excluded_cols and res:
                try:
                    resource_id = resource_label_to_index[col_name]
                    resources[resource_id] = float(res)
                except ValueError:
                    raise ValueError(f"Invalid resource value '{res}' for task {row['ID']} in resource column {col_name}. Expected a numeric value.")

        task = Task(i, row['ID'], row['Name'], row['Duration'], predecessors, external_predecessors, successors, resources, project, row['Section'])
        adjust_task_dates_by_offset(task, project.start_offset)
        tasks.append(task)

    resources_list = [Resource(i, row['ID'], row['Name'], row['Type'], row['Units']) for i, row in resources_df.iterrows()]

    # Check if any task demands more units of a resource than its total capacity
    resources_availability = {resource.id: resource.units for resource in resources_list}
    for task in tasks:
        for resource_id, units in task.resources.items():
            if units > resources_availability[resource_id]:
                resource_label = resources_list[resource_id].name
                raise ValueError(f"Project {instanceName} - Task {task.label} {task.name} demands {units} units of resource {resource_label}, but only {resources_availability[resource_id]} units are available.")

    project.tasks.extend(tasks)
    project.resources.extend(resources_list)
    
    return project



def combine_projects(portfolio):
    """
    Combine all projects in a portfolio into a single project, preserving the task.project reference for each task.
    
    Args:
        portfolio (Portfolio): The portfolio containing multiple projects.
        
    Returns:
        Project: A single combined project.
    """
    combined_project = Project(
        instanceName="CombinedProject",
        startDate=portfolio.start_date,
        deadline="",
        dailyPenalty=0
    )

    task_offset = 0
    for project in portfolio.projects:
        project_task_ids = [task.id for task in project.tasks]  # Collecting the task IDs for each project
        for task in project.tasks:
            cloned_task = copy.deepcopy(task)
            cloned_task.id += task_offset
            
            # Adjusting predecessors and successors to maintain the dictionary structure
            cloned_task.predecessors = {pred_id + task_offset: gap for pred_id, gap in task.predecessors.items() if pred_id in project_task_ids}
            cloned_task.successors = {succ_id + task_offset: gap for succ_id, gap in task.successors.items() if succ_id in project_task_ids}
            
            combined_project.tasks.append(cloned_task)
        
        task_offset += len(project.tasks)
        
        # Add resources only from the first project to avoid duplication
        if combined_project.resources == []:
            combined_project.resources.extend([copy.deepcopy(resource) for resource in project.resources])
    
    return combined_project


def decompose_project(combined_project, original_projects):
    """
    Decompose a combined project back into individual projects based on the task.project attribute.
    
    Args:
        combined_project (Project): The combined project after processing.
        original_projects (list): List of original projects for reference.
        
    Returns:
        list: List of decomposed projects.
    """
    decomposed_tasks = {project.instanceName: [] for project in original_projects}
    
    # Assign tasks back to their original projects using the task.project attribute
    for task in combined_project.tasks:
        decomposed_tasks[task.project.instanceName].append(task)
    
    decomposed_projects = []
    for project in original_projects:
        decomposed_project = Project(
            instanceName=project.instanceName,
            startDate=project.startDate,
            deadline=project.deadline,
            dailyPenalty=project.dailyPenalty,
            dailyWaterConsumption=project.dailyWaterConsumption
        )
        decomposed_project.tasks = decomposed_tasks[project.instanceName]
        decomposed_project.resources = project.resources  # Reuse the original resources
        decomposed_projects.append(decomposed_project)
    
    return decomposed_projects

def log_project_penalty(project):
    """Logs the penalty details of a project to the specified log file using the logging module.
    
    Args:
        project (Project): The project for which penalty details need to be logged.
    """
    penalty = project.compute_penalty()
    delivery_date = project.get_delivery_date()
    
    if penalty > 0:
        report = (f"- Project '{project.instanceName}' was delivered late by "
                  f"{get_labor_days_difference(datetime.strptime(project.deadline, '%d-%m-%Y').date(), datetime.strptime(delivery_date, '%d-%m-%Y').date())} "
                  f"labor days, on {delivery_date}. The deadline was {project.deadline}. "
                  f"The total penalty is {penalty:,.1f} k$.")
    else:
        report = f"- Project '{project.instanceName}' was delivered on time on {delivery_date}. There is no penalty."
    
    # Log the report using the logging module
    logging.info(report)


def log_construction_duration_and_water_consumption(project):
    """
    Log the initial date, end date, and duration in labor days for the "Construction" section of a project.
    
    Args:
        project (Project): The project instance containing tasks.
        
    Returns:
        tuple: A tuple containing the initial date, end date, and duration in labor days for the "Construction" section.
    """

    # Filter tasks that belong to the "Construction" section
    construction_tasks = [task for task in project.tasks if task.section == "Construction"]
    
    # If there are no construction tasks, return an appropriate message
    if not construction_tasks:
        logging.info(f"- Project '{project.instanceName}': No tasks found for the 'Construction' section in the project.")
        return None, None, 0
    
    # Find the earliest start date and the latest finish date among the construction tasks
    initial_date = min([datetime.strptime(task.start_date, "%d-%m-%Y") for task in construction_tasks])
    end_date = max([datetime.strptime(task.finish_date, "%d-%m-%Y") for task in construction_tasks])
    
    # Calculate the construction duration in labor days and associated water consumption
    construction_duration = get_labor_days_difference(initial_date, end_date)
    water_consumption = construction_duration * project.dailyWaterConsumption

    # Log the results
    report = (f"- Project '{project.instanceName}' Construction phase: "
        f"Initial date: {initial_date.strftime('%d-%m-%Y')}, End date: {end_date.strftime('%d-%m-%Y')}. Duration: {construction_duration} labor days. "
        f"The total water consumption in that phase has been {water_consumption:,.0f} m3.")
    
    logging.info(report)
    
    return initial_date, end_date, construction_duration


def get_external_predecessor_notation(project_name, task_label, lag):
    """
    Get the notation for an external predecessor task, including any extra time.

    Args:
        project_name (str): The name of the project the task belongs to.
        task_label (str): The label of the predecessor task.
        lag (int): The extra time for the predecessor task.

    Returns:
        str: The notation for the external predecessor task with extra time.
    """
    if lag == 0:
        return f"{project_name}-{task_label}"
    elif lag > 0:
        return f"{project_name}-{task_label}FC+{lag}"
    else:
        return f"{project_name}-{task_label}FC{lag}"  # since lag is negative, no need for additional '-'

def project_to_df(project):
    """
    Convert project task data into a pandas DataFrame.

    Args:
        project (Project): The project whose tasks' data needs to be converted.

    Returns:
        pd.DataFrame: A DataFrame containing project task data.
    """
    # Create a mapping from task ID to task object for quick lookups
    task_map = {task.id: task for task in project.tasks}

    # Create a DataFrame to store the project tasks
    data = {
        "Task Label": [task.label for task in project.tasks],
        "Section": [task.section for task in project.tasks],
        "Task Name": [task.name for task in project.tasks],
        "Duration": [task.duration for task in project.tasks],
        "Start Date": [task.start_date for task in project.tasks],
        "Finish Date": [task.finish_date for task in project.tasks],
        "Start Time": [task.start_time for task in project.tasks],
        "Finish Time": [task.finish_time for task in project.tasks],
    }

    # Collect predecessors and successors data
    predecessors_data = []
    successors_data = []
    for task in project.tasks:
        # Use the task_map to look up tasks by ID
        pred_data = [get_predecessor_notation(task_map[pred].label, lag) for pred, lag in task.predecessors.items() if pred in task_map]
        # Split the combined project_name-label string into separate project_name and task_label
        ext_pred_data = [get_external_predecessor_notation(project_name_label.split("-")[0], project_name_label.split("-")[1], lag) for project_name_label, lag in task.external_predecessors.items()]
        pred_data.extend(ext_pred_data)
        succ_data = [get_predecessor_notation(task_map[succ].label, lag) for succ, lag in task.successors.items() if succ in task_map]
        predecessors_data.append("; ".join(pred_data))
        successors_data.append("; ".join(succ_data))

    # Add predecessors and successors data to the DataFrame
    data["Predecessors"] = predecessors_data
    data["Successors"] = successors_data

    df = pd.DataFrame(data)
    return df






def write_solutions_to_excel(dfs, sheet_names):
    """
    Write multiple DataFrames to an Excel file with different sheets.

    Args:
        dfs (list): List of pandas DataFrames to be written to the Excel file.
        sheet_names (list): List of sheet names corresponding to each DataFrame.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Write all dataframes to a single Excel file with different sheets
    output_filename = f"{PORTFOLIO_FILE.split('.')[0]}_solutions.xlsx"
    with pd.ExcelWriter(os.path.join(OUTPUTS_DIR, output_filename)) as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def get_predecessor_notation(task_label, lag):
    """
    Get the notation for a predecessor task, including any extra time.

    Args:
        task_label (str): The label of the predecessor task.
        lag (int): The extra time for the predecessor task.

    Returns:
        str: The notation for the predecessor task with extra time.
    """
    if lag == 0:
        return task_label
    elif lag > 0:
        return f"{task_label}FC+{lag}"
    else:
        return f"{task_label}FC-{lag}"


def topological_sort(tasks):
    """
    Perform a topological sort on a list of tasks.

    Args:
        tasks (list): List of Task objects.

    Returns:
        list: A list of tasks in topological order.
    """
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


def TORA_Heuristic(project):
    solution = Solution()
    """
    Apply the Topological Ordering and Resource Allocation (TORA) heuristic to a project.

    Args:
        project (Project): The project on which to apply the TORA heuristic.

    Returns:
        Solution: The solution generated by the TORA heuristic.
    """
    logging.info("Delays in tasks due to collision of resources:")
    
    # Initialize resources availability
    resources_availability = {resource.id: resource.units for resource in project.resources}

    # Initialize dictionary to keep track of resources used by each task
    resources_used = {task.id: set() for task in project.tasks}

    # Check if any task demands more units of a resource than its total capacity
    for task in project.tasks:
        for resource_id, units in task.resources.items():
            if units > resources_availability[resource_id]:
                sys.exit(f"[ERROR]: Task {task.label} demands {units} units of resource {resource_id}, but only {resources_availability[resource_id]} units are available.")

    # Topologically sort the tasks. A topological sort is an algorithm that takes a directed
    # graph and returns a linear ordering of its vertices (nodes) such that, for every
    # directed edge (u, v) from vertex u to vertex v, u comes before v in the ordering
    sorted_tasks = topological_sort(project.tasks)

    # Loop over the tasks in topological order
    for task in sorted_tasks:
        # Calculate earliest start time for task considering predecessor dependencies
        earliest_start_time = task.start_time # Use the task's current start time as the base
        for pred_id, lag in task.predecessors.items():
            pred_task = project.tasks[pred_id]
            if lag >= 0:
                # Add extra time to the ending time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.finish_time + lag)
            else:
                # Add extra time to the starting time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.start_time + abs(lag))

        # Refine earliest start time for task considering resource availability
        for resource_id, units in task.resources.items():
            initial_earliest_start_time = earliest_start_time
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
            
            delay = earliest_start_time - initial_earliest_start_time
            if delay > 0:
                logging.info(f"- Task '{task.label}' from project '{task.project.instanceName}' has been delayed by {delay} days due to insufficient availability of resource '{project.resources[resource_id].label}'.")

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
    
    # Sort tasks by label, for better intrepretation at the output
    solution.tasks.sort(key=lambda task: int(task.id))

    solution.time = task.finish_time
    return solution


def network_diagram(project):
    """
    Generate a network diagram for a project based on a topological sort.

    Args:
        project (Project): The project for which to generate the network diagram.

    Returns:
        Solution: The solution representing the network diagram.
    """
    solution = Solution()

    # Topologically sort the tasks. A topological sort is an algorithm that takes a directed
    # graph and returns a linear ordering of its vertices (nodes) such that, for every
    # directed edge (u, v) from vertex u to vertex v, u comes before v in the ordering
    sorted_tasks = topological_sort(project.tasks)

    # Loop over the tasks in topological order
    for task in sorted_tasks:
        # Calculate earliest start time for task considering predecessor dependencies
        earliest_start_time = task.start_time # Use the task's current start time as the base
        for pred_id, lag in task.predecessors.items():
            pred_task = project.tasks[pred_id]
            if lag >= 0:
                # Add extra time to the ending time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.finish_time + lag)
            else:
                # Add extra time to the starting time of predecessor
                earliest_start_time = max(earliest_start_time, pred_task.start_time + abs(lag))
        
        # Update finish time of task
        task.start_time = earliest_start_time
        task.finish_time = earliest_start_time + task.duration

        # Add task to project schedule
        solution.tasks.append(task)
 
    # Sort tasks by label, for better intrepretation at the output
    solution.tasks.sort(key=lambda task: int(task.id))
    solution.time = task.finish_time
    return solution


def display_gantt_chart(project, label=None, save_to_file=True):

    # Prepare data for the Gantt chart
    tasks_data = [{
        "Task": task.name,
        "Task_ID": task.id,
        "Start": datetime.strptime(task.start_date, "%d-%m-%Y"),
        "Finish": datetime.strptime(task.finish_date, "%d-%m-%Y"),
        "Duration": task.duration,
    } for task in project.tasks]
    
    # Sort tasks_data by task_id
    tasks_data.sort(key=lambda x: x["Task_ID"])

    # Reverse the order of tasks_data
    tasks_data.reverse()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, task in enumerate(tasks_data):
        start_date = task["Start"]
        finish_date = task["Finish"]
        duration = (finish_date - start_date).days
        if duration == 0:  # Milestone task
            ax.plot(start_date, idx, 'dk', markersize=5) 
        else:
            ax.barh(idx, duration, left=start_date, color="lightblue")

    # Set y-axis ticks and labels
    y_labels = [task["Task"] for task in tasks_data]
    y_ticks = range(len(y_labels))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Set labels and title with larger font size
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Task")
    ax.set_title(project.instanceName, fontsize=16)  # Set larger title font size

    # Format the x-axis to display dates correctly
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())  
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

    # Add vertical dotted grid lines
    ax.xaxis.grid(True, linestyle='dotted')

    if label:
        handles, labels = ax.get_legend_handles_labels()
        patch = plt.Line2D([0], [0], markerfacecolor="lightblue", marker='s', markersize=10, color="white", label=label)
        handles.append(patch) 
        ax.legend(handles=handles, loc="upper right")

    # Show the plot
    plt.tight_layout()

    # Save the plot to a file
    if save_to_file:
        # Construct filename using project's instanceName and label (if provided)
        base_name = project.instanceName
        if label:
            base_name += f" {label}"  # Add a space before the label
        filename = "".join([c for c in base_name if c.isalpha() or c.isdigit() or c == ' ' or c == '_']).rstrip() + ".png"
        
        # Ensure the /png subfolder exists
        png_folder = os.path.join(OUTPUTS_DIR, 'png')
        os.makedirs(png_folder, exist_ok=True)  # This will create the folder if it doesn't exist
        
        filepath = os.path.join(png_folder, filename)
        plt.savefig(filepath, dpi=300)
        print(f"Gantt chart saved to: {filepath}")
        plt.close(fig)  # Close the figure after saving
    else:
        # Show the plot on screen if not saving to file
        plt.show(block=False)



def adjust_external_predecessors_and_successors(tasks):
    """
    Adjust the external predecessors for a list of tasks.
    
    For each task in the provided list, this function checks its external predecessors
    and adjusts its 'predecessors' attribute to include the identified tasks. It also
    ensures that each identified predecessor task lists the current task as a successor.
    
    The function assumes that external predecessors are represented in the format 
    "project_name-label". The function uses this format to identify the corresponding
    task ID and make the necessary adjustments.
    
    Args:
        tasks (list): A list of Task objects. Each Task object should have 'external_predecessors',
                      'predecessors', and 'successors' attributes.
    
    Returns:
        None. The function modifies the 'predecessors' and 'successors' attributes of the provided tasks in-place.
    """
    # Create a mapping from project-label to task ID
    project_label_to_id = {(task.project.instanceName, task.label): task.id for task in tasks}
    
    # Iterate through tasks and adjust external predecessors
    for task in tasks:
        for ext_pred_key, lag in task.external_predecessors.items():
            project_name, label = ext_pred_key.split('-')
            corresponding_task_id = project_label_to_id.get((project_name, label))
            if corresponding_task_id is not None:
                # Add to the predecessors attribute without removing it from external_predecessors
                task.predecessors[corresponding_task_id] = lag
                
                # Add the current task as a successor to the corresponding predecessor task
                pred_task = tasks[corresponding_task_id]
                pred_task.successors[task.id] = lag



def check_task_consistency(project):
    """
    Check the consistency of predecessors and successors declarations in a project's tasks.
    
    Args:
        project (Project): The project instance with tasks to be checked.
        
    Raises:
        ValueError: If there's an inconsistency between the predecessors and successors.
    """
    for task in project.tasks:
        for successor_id in task.successors:
            successor_task = project.tasks[successor_id]
            if task.id not in successor_task.predecessors:
                raise ValueError(f"Inconsistency detected in {project.instanceName}: Task {task.label} lists Task {successor_task.label} as its successor, but Task {successor_task.label} does not list Task {task.label} as its predecessor.")
        for predecessor_id in task.predecessors:
            predecessor_task = project.tasks[predecessor_id]
            if task.id not in predecessor_task.successors:
                raise ValueError(f"Inconsistency detected in {project.instanceName}: Task {task.label} lists Task {predecessor_task.label} as its predecessor, but Task {predecessor_task.label} does not list Task {task.label} as its successor.")
                
    print(f"No inconsistencies detected in task predecessors and successors for {project.instanceName}.")




def report_with_chatgpt(input_filename, output_filename, model="gpt-3.5-turbo"):

    with open(input_filename, 'r') as file:
        text = file.read()

    prompt = (
        "I have this report from my Project Management System. The report compares the same projects under two scenarios: "
        "constrained projects means that resources are shared and may induce delays, and not constrained means that resources "
        "collisions haven't been taking into account when building schedules. Please write a report from these data, as if "
        "you were a Portfolio manager. Take into account that resource-constrained and not constrained "
        "are the same projects (P1, P2..) but under constraints or not, so you can compare P1 constrained vs not constrained, "
        "because it's the same project. Please give your insights about which resource should be increased for "
        "maximum benefit. This is the report: " 
        + text
    )

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1,
    )

    with open(output_filename, 'w') as file:
        file.write(response.choices[0].message["content"])


