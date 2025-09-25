import argparse
import os
import numpy as np
import symnmf_c  # Import C module

GOALS = ['symnmf', 'sym', 'ddg', 'norm']
# Set random seed at the beginning of the code
np.random.seed(1234)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run symnmf using C extension")
    parser.add_argument('k', type=int, help='Number of clusters')
    parser.add_argument('goal', type=str, choices=GOALS, help='Goal')
    parser.add_argument('file_name', type=str, help='Input file path')
    return parser.parse_args()

def read_data_points(file_name):
    """Read data points from file and convert to numpy array"""
    points = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Convert line to list of floats
            row = [float(x) for x in line.strip().replace(',', ' ').split() if x]
            if row:
                points.append(row)
    return np.array(points)

def initialize_H(W, k):
    m = np.mean(W)
    n = W.shape[0]
    upper = 2.0 * np.sqrt(m / k)
    H = np.random.uniform(0.0, upper, size=(n, k))
    return H

def handle_goal(goal, X, k):
    """Call appropriate C function based on goal"""
    if goal == 'symnmf':
        W = symnmf_c.norm(X)
        H = initialize_H(W, k)
        return symnmf_c.symnmf(H, W)
    elif goal == 'sym':
        return symnmf_c.sym(X)
    elif goal == 'ddg':
        return symnmf_c.ddg(X)
    elif goal == 'norm':
        return symnmf_c.norm(X)
    else:
        raise ValueError("Invalid goal")

def print_matrix(result):
    """Output the required matrix separated by comma, each row in a line"""
    for row in result:
        print(','.join(f'{val:.4f}' for val in row))

def main():
    """Main execution function"""
    try:
        args = parse_arguments()
        assert args.goal in ['symnmf', 'sym', 'ddg', 'norm'], "Invalid goal"
        assert os.path.exists(args.file_name), "File does not exist"
        X = read_data_points(args.file_name)
        if args.goal == 'symnmf':
            assert 1 <= args.k < X.size, "k must be greater than 1 and less than number of data points"
        result = handle_goal(args.goal, X, args.k)
        print_matrix(result)
    except Exception as e:
        print(f"An Error Has Occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()
