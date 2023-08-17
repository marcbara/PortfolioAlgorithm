from utils import readProjects, readInputs, TORA_Heuristic, network_diagram, SolutionToDF, write_solutions_to_excel
import copy


def main():
    # Read projects from the file
    projects = readProjects()

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    for project in projects:
        # Read inputs for one project
        original_inputs = readInputs(project)

        # Make a deep copy for TORA
        inputs_for_tora = copy.deepcopy(original_inputs)
        # Calculate solution for the given scenario
        solution_constrained = TORA_Heuristic(inputs_for_tora)
        df_constrained = SolutionToDF(inputs_for_tora, solution_constrained)
        dfs.append(df_constrained)
        sheet_names.append(project.instanceName + "_Constrained")

        # Make another deep copy for the network diagram
        inputs_for_nd = copy.deepcopy(original_inputs)
        # Calculate solution only as a network diagram
        solution_nd = network_diagram(inputs_for_nd)
        df_not_constrained = SolutionToDF(inputs_for_nd, solution_nd)
        dfs.append(df_not_constrained)
        sheet_names.append(project.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

if __name__ == "__main__":
    main()

