from utils import readTests, readInputs, TORA_Heuristic, printSolutionToExcel, network_diagram
from classes import Test

def main():
    # Read tests from the file
    tests = readTests("test2run.txt")

    for test in tests:
        # Read inputs for the test inputs
        inputs = readInputs(test.instanceName)

        # Calculate solution for the given scenario
        solution = TORA_Heuristic(inputs)
        printSolutionToExcel(inputs, solution, test.instanceName + "_Constrained")
        
        # Calculate solution only as a network diagram
        solution_nd = network_diagram(inputs)
        printSolutionToExcel(inputs, solution_nd, test.instanceName + "_notConstrained")

if __name__ == "__main__":
    main()