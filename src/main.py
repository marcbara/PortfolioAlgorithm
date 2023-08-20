from utils import readProjects, readInputs, TORA_Heuristic, network_diagram, ProjectToDF, write_solutions_to_excel, set_project_task_dates, combine_projects, decompose_project
import copy


def main_independentprojects():
    # Read projects from the file
    portfolio = readProjects()

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    for project in portfolio.projects:
        # Read inputs (tasks and resources) for one project
        original_project = readInputs(project)
        
        # Make a deep copy for TORA
        project_for_tora = copy.deepcopy(original_project)
        # Calculate solution for the given scenario
        solution_constrained = TORA_Heuristic(project_for_tora)
        solved_constrained_project = solution_constrained.to_project(project_for_tora)
        set_project_task_dates(solved_constrained_project, portfolio.start_date)

        df_constrained = ProjectToDF(solved_constrained_project)
        dfs.append(df_constrained)
        sheet_names.append(project.instanceName + "_Constrained")

        # Make another deep copy for the network diagram
        project_for_nd = copy.deepcopy(original_project)
        # Calculate solution only as a network diagram
        solution_nd = network_diagram(project_for_nd)
        solved_notconstrained_project = solution_nd.to_project(project_for_nd)
        set_project_task_dates(solved_notconstrained_project, portfolio.start_date)

        df_not_constrained = ProjectToDF(solved_notconstrained_project)
        dfs.append(df_not_constrained)
        sheet_names.append(project.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

def main_joinprojects():
    # Read projects from the file
    portfolio = readProjects()

    # Read inputs (tasks and resources) for each project in the portfolio
    for project in portfolio.projects:
        readInputs(project)

    # Combine all projects into one
    combined_project = combine_projects(portfolio)

    # Process the combined project
    original_combined_project = copy.deepcopy(combined_project)

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    # Process the combined project with TORA
    project_for_tora = copy.deepcopy(original_combined_project)
    solution_constrained = TORA_Heuristic(project_for_tora)
    solved_constrained_project = solution_constrained.to_project(project_for_tora)

    # Decompose the TORA-processed combined project and set dates
    decomposed_projects_constrained = decompose_project(solved_constrained_project, portfolio.projects)
    for project in decomposed_projects_constrained:
        set_project_task_dates(project, portfolio.start_date)

    # Process the combined project with the network diagram
    project_for_nd = copy.deepcopy(original_combined_project)
    solution_nd = network_diagram(project_for_nd)
    solved_notconstrained_project = solution_nd.to_project(project_for_nd)

    # Decompose the network diagram-processed combined project and set dates
    decomposed_projects_notconstrained = decompose_project(solved_notconstrained_project, portfolio.projects)
    for project in decomposed_projects_notconstrained:
        set_project_task_dates(project, portfolio.start_date)

    # Convert each decomposed project to a dataframe and add to the lists
    for project in decomposed_projects_constrained:
        df_constrained = ProjectToDF(project)
        dfs.append(df_constrained)
        sheet_names.append(project.instanceName + "_Constrained")
    
    for project in decomposed_projects_notconstrained:
        df_not_constrained = ProjectToDF(project)
        dfs.append(df_not_constrained)
        sheet_names.append(project.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)


if __name__ == "__main__":
    main_joinprojects()