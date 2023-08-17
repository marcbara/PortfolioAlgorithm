from utils import readTests, readInputs, TORA_Heuristic, printSolutionToExcel, network_diagram, OUTPUTS_DIR
from classes import Test
import pandas as pd
import os


def main():
# Read tests from the file
    tests = readTests()

    # Lists to accumulate dataframes and sheet names
    dfs = []
    sheet_names = []

    for test in tests:
        # Read inputs for the test inputs
        inputs = readInputs(test.instanceName)

        # Calculate solution for the given scenario
        solution_df_constrained = TORA_Heuristic(inputs)
        dfs.append(printSolutionToExcel(inputs, solution_df_constrained, test.instanceName + "_Constrained"))
        sheet_names.append(test.instanceName + "_Constrained")

        # Calculate solution only as a network diagram
        solution_nd = network_diagram(inputs)
        dfs.append(printSolutionToExcel(inputs, solution_nd, test.instanceName + "_notConstrained"))
        sheet_names.append(test.instanceName + "_notConstrained")

    # Write all dataframes to a single Excel file with different sheets
    with pd.ExcelWriter(os.path.join(OUTPUTS_DIR, "Portfolio_solutions.xlsx")) as writer:
        for df, sheet_name in zip(dfs, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":
    main()
