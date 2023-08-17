from utils import readTests, readInputs, TORA_Heuristic, network_diagram, SolutionToDF, write_solutions_to_excel

def main():
    # Read tests from the file
    tests = readTests()

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    for test in tests:
        # Read inputs for the test
        inputs = readInputs(test.instanceName)

        # Calculate solution for the given scenario
        solution_constrained = TORA_Heuristic(inputs)
        df_constrained = SolutionToDF(inputs, solution_constrained)
        dfs.append(df_constrained)
        sheet_names.append(test.instanceName + "_Constrained")

        # Calculate solution only as a network diagram
        solution_nd = network_diagram(inputs)
        df_not_constrained = SolutionToDF(inputs, solution_nd)
        dfs.append(df_not_constrained)
        sheet_names.append(test.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    write_solutions_to_excel(dfs, sheet_names)

if __name__ == "__main__":
    main()
