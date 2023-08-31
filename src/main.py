from utils import read_projects, read_inputs 
from utils import TORA_Heuristic, network_diagram, set_project_task_dates
from utils import combine_projects, decompose_project
from utils import display_gantt_chart, project_to_df, write_solutions_to_excel, log_project_penalty, adjust_external_predecessors_and_successors
from utils import check_task_consistency, log_construction_duration_and_water_consumption
from utils import report_with_chatgpt, log_filename, AI_insights_filename, read_secret_files
from utils import generate_ai_insights, generate_gantts
import copy
import logging
import matplotlib.pyplot as plt
import os

def main_independentprojects():
    """
    The function `main_independentprojects()` reads multiple projects from a file, processes them with TORA
    (Task Ordering and Resource Allocation) and Network Diagram algorithms, converts the resulting projects
    into dataframes and saves them in an excel file with different sheets. It does this for each project independently.

    Parameters: None

    Returns: None
    """
    os.system('cls')

    # Read projects from the file
    portfolio = read_projects()

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    for project in portfolio.projects:
        # Read inputs (tasks and resources) for one project
        original_project = read_inputs(project)
        try:
            check_task_consistency(original_project)
        except ValueError as e:
            print("Error:", e)
            exit(1)
        
        # Make a deep copy for TORA
        project_for_tora = copy.deepcopy(original_project)

        # Calculate solution for the given scenario
        solution_constrained = TORA_Heuristic(project_for_tora)
        solved_constrained_project = solution_constrained.to_project(project_for_tora)
        set_project_task_dates(solved_constrained_project, portfolio.start_date)

        df_constrained = project_to_df(solved_constrained_project)
        dfs.append(df_constrained)
        sheet_names.append(project.instanceName + "_Constrained")

        # Make another deep copy for the network diagram
        project_for_nd = copy.deepcopy(original_project)
        # Calculate solution only as a network diagram
        solution_nd = network_diagram(project_for_nd)
        solved_notconstrained_project = solution_nd.to_project(project_for_nd)
        set_project_task_dates(solved_notconstrained_project, portfolio.start_date)

        df_not_constrained = project_to_df(solved_notconstrained_project)
        dfs.append(df_not_constrained)
        sheet_names.append(project.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

def main_jointprojects():
    """
    The function `main_joinprojects()` reads input projects from a file, combines them, processes them with
    TORA (Task Ordering and Resource Allocation), Network Diagram algorithm and generates decomposed
    output projects using specialized functions. It then converts the decomposed projects into dataframes,
    logs possible penalties and water consumption, displays Gantt charts of the resources and saves the dataframes in an excel
    file with different sheets.

    Parameters: None

    Returns: None
    """
    os.system('cls')

    # Read projects from the file
    portfolio = read_projects()

    # Read inputs (tasks and resources) for each project in the portfolio
    for project in portfolio.projects:
        read_inputs(project)
        try:
            check_task_consistency(project)
        except ValueError as e:
            print("Error:", e)
            exit(1)

    # Combine all projects into one
    combined_project = combine_projects(portfolio)

    # Process the combined project
    original_combined_project = copy.deepcopy(combined_project)

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    # Process the combined project with TORA
    project_for_tora = copy.deepcopy(original_combined_project)
    adjust_external_predecessors_and_successors(project_for_tora.tasks)
    solution_constrained = TORA_Heuristic(project_for_tora)
    solved_constrained_project = solution_constrained.to_project(project_for_tora)

    # Decompose the TORA-processed combined project and set dates
    decomposed_projects_constrained = decompose_project(solved_constrained_project, portfolio.projects)
    for project in decomposed_projects_constrained:
        set_project_task_dates(project, portfolio.start_date)

    # Process the combined project with the network diagram
    project_for_nd = copy.deepcopy(original_combined_project)
    adjust_external_predecessors_and_successors(project_for_nd.tasks)
    solution_nd = network_diagram(project_for_nd)
    solved_notconstrained_project = solution_nd.to_project(project_for_nd)

    # Decompose the network diagram-processed combined project and set dates
    decomposed_projects_notconstrained = decompose_project(solved_notconstrained_project, portfolio.projects)
    for project in decomposed_projects_notconstrained:
        set_project_task_dates(project, portfolio.start_date)

    # Convert each decomposed project to a dataframe and add to the lists
    logging.info("\nReport of Projects when the availability of resources is a constraint:")
    for project in decomposed_projects_constrained:
        df_constrained = project_to_df(project)
        dfs.append(df_constrained)
        log_project_penalty(project)
        log_construction_duration_and_water_consumption(project)
        sheet_names.append(project.instanceName + "_Constrained")
        if generate_gantts:
            display_gantt_chart(project, "Constrained Resources")

    
    logging.info("\nReport of Projects not constrained by resources (as if resources were not limiting):")
    for project in decomposed_projects_notconstrained:
        df_not_constrained = project_to_df(project)
        dfs.append(df_not_constrained)
        log_project_penalty(project)
        log_construction_duration_and_water_consumption(project)
        sheet_names.append(project.instanceName + "_notConstrained")
        if generate_gantts:
            display_gantt_chart(project, "Not-Constrained Resources")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

    # Report with ChatGPT
    if read_secret_files and generate_ai_insights:
        report_with_chatgpt(log_filename, AI_insights_filename)

    # Block Gantt Charts until user closes them
    plt.show()

if __name__ == "__main__":
    main_jointprojects()