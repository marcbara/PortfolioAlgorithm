from utils import readTests, readInputs, TORA_Heuristic, printSolutionToExcel, network_diagram
from classes import Test

def main():
    # Read tests from the file
    tests = read_tests("test2run.txt")

    for test in tests:
        # Read inputs for the test inputs
        inputs = read_inputs(test.instanceName)

        # Calculate solution for the given scenario
        solution = TORA_Heuristic(inputs)
        print_solution_to_excel(solution, test.instanceName + "_Constrained")
        
        # Calculate solution only as a network diagram
        solution_nd = network_diagram(inputs)
        print_solution_to_excel(solution_nd, test.instanceName + "_notConstrained")

if __name__ == "__main__":
    main()