from utils import readProjects, readInputs, TORA_Heuristic, network_diagram, ProjectToDF, write_solutions_to_excel
import copy


def main():
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

        df_constrained = ProjectToDF(solved_constrained_project)
        dfs.append(df_constrained)
        sheet_names.append(project.instanceName + "_Constrained")

        # Make another deep copy for the network diagram
        project_for_nd = copy.deepcopy(original_project)
        # Calculate solution only as a network diagram
        solution_nd = network_diagram(project_for_nd)
        solved_notconstrained_project = solution_nd.to_project(project_for_nd)

        df_not_constrained = ProjectToDF(solved_notconstrained_project)
        dfs.append(df_not_constrained)
        sheet_names.append(project.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

if __name__ == "__main__":
    main()

