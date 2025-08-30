import argparse
import numpy as np
import symnmf  # Import C module

# Set random seed at the beginning of the code
np.random.seed(1234)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run symnmf using C extension")
    parser.add_argument('k', type=int, help='Number of clusters')
    parser.add_argument('goal', type=str, choices=['symnmf', 'sym', 'ddg', 'norm'], help='Goal')
    parser.add_argument('file_name', type=str, help='Input file path')
    return parser.parse_args()

def read_data_points(file_name):
    """Read data points from file and convert to numpy array"""
    points = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = [float(x) for x in line.strip().replace(',', ' ').split() if x]
            if row:
                points.append(row)
    return np.array(points)

def handle_goal(goal, X, k):
    """Call appropriate C function based on goal"""
    if goal == 'symnmf':
        # For symnmf, we need to pass initial H, W and other arguments
        return symnmf.symnmf(X, k)
    elif goal == 'sym':
        return symnmf.sym(X)
    elif goal == 'ddg':
        return symnmf.ddg(X)
    elif goal == 'norm':
        return symnmf.norm(X)

def print_matrix(result):
    """Output the required matrix separated by comma, each row in a line"""
    for row in result:
        print(','.join(f'{val:.4f}' for val in row))

def main():
    """Main execution function"""
    try:
        args = parse_arguments()
        X = read_data_points(args.file_name)
        result = handle_goal(args.goal, X, args.k)
        print_matrix(result)
    except Exception:
        print("An Error Has Occurred")
        exit(1)

if __name__ == "__main__":
    main()
