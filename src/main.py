from utils import readProjects, readInputs, TORA_Heuristic, network_diagram, SolutionToDF, write_solutions_to_excel

def main():
    # Read projects from the file
    projects = readProjects()

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    for project in projects:
        # Read inputs for one project
        inputs = readInputs(project.instanceName)

        # Calculate solution for the given scenario
        solution_constrained = TORA_Heuristic(inputs)
        df_constrained = SolutionToDF(inputs, solution_constrained)
        dfs.append(df_constrained)
        sheet_names.append(project.instanceName + "_Constrained")

        # Calculate solution only as a network diagram
        solution_nd = network_diagram(inputs)
        df_not_constrained = SolutionToDF(inputs, solution_nd)
        dfs.append(df_not_constrained)
        sheet_names.append(project.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

if __name__ == "__main__":
    main()
