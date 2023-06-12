import numpy as np
import matplotlib.pyplot as plt
from random import sample

"""
Sudoku board initializer
Credit: https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
"""
def generate(n, num_clues):
    # Generate a sudoku problem of order n with "num_clues" cells assigned
    # Return dictionary containing clue cell indices and corresponding values
    # (You do not need to worry about components inside returned dictionary)
    N = range(n)

    rows = [g*n+r for g in sample(N,n) for r in sample(N,n)]
    cols = [g*n+c for g in sample(N,n) for c in sample(N,n)]
    nums = sample(range(1,n**2+1), n**2)

    S = np.array([[nums[(n*(r%n)+r//n+c)%(n**2)] for c in cols] for r in rows])
    indices = sample(range(n**4), num_clues)
    values = S.flatten()[indices]

    mask = np.full((n**2, n**4), True)
    mask[:, indices] = False
    i, j = np.unravel_index(indices, (n**2,n**2))

    for c in range(num_clues):
        v = values[c]-1
        maskv = np.full((n**2, n**2), True)
        maskv[i[c]] = False
        maskv[:,j[c]] = False
        maskv[(i[c]//n)*n:(i[c]//n)*n+n,(j[c]//n)*n:(j[c]//n)*n+n] = False
        mask[v] = mask[v] * maskv.flatten()

    return {'n':n, 'indices':indices, 'values':values, 'valid_indices':mask}


def display(problem):
    # Display the initial board with clues filled in (all other cells are 0)
    n = problem['n']
    empty_board = np.zeros(n**4, dtype=int)
    empty_board[problem['indices']] = problem['values']
    print("Sudoku puzzle:\n", np.reshape(empty_board, (n**2,n**2)), "\n")


def initialize(problem):
    # Returns a random initial sudoku board given problem
    n = problem['n']
    S = np.zeros(n**4, dtype=int)
    S[problem['indices']] = problem['values']

    all_values = list(np.repeat(range(1,n**2+1), n**2))
    for v in problem['values']:
        all_values.remove(v)
    all_values = np.array(all_values)
    np.random.shuffle(all_values)

    indices = [i for i in range(S.size) if i not in problem['indices']]
    S[indices] = all_values
    S = S.reshape((n**2,n**2))

    return S


def successors(S, problem):
    # Returns list of all successor states of S by swapping two non-clue entries
    mask = problem['valid_indices']
    indices = [i for i in range(S.size) if i not in problem['indices']]
    succ = []

    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            s = np.copy(S).flatten()
            if s[indices[i]] == s[indices[j]]: continue
            if not (mask[s[indices[i]]-1, indices[j]] and mask[s[indices[j]]-1, indices[i]]): continue
            s[indices[i]], s[indices[j]] = s[indices[j]], s[indices[i]]
            succ.append(s.reshape(S.shape))

    return succ


"""
WRITE THIS FUNCTION
"""
def num_errors(S):
    # Given a current sudoku board state (2d NumPy array), compute and return total number of errors
    # Count total number of missing numbers from each row, column, and non-overlapping square blocks
    total = 0
    # number of errors in all rows
    for i in range(S.shape[0]):
        existed = set()
        repeated = [x for x in S[i] if x in existed or existed.add(x)]
        total += len(repeated)

    # number of errors in all columns
    for i in range(S.shape[0]):
        existed = set()
        repeated = [x for x in S.T[i] if x in existed or existed.add(x)]
        total += len(repeated)

    # number of errors in all squares
    for row in range(0, S.shape[0], int(S.shape[0] ** 0.5)):
        for col in range(0, S.shape[0], int(S.shape[0] ** 0.5)):
            roi = S[row:row + int(S.shape[0] ** 0.5), col:col + int(S.shape[0] ** 0.5)]
            existed = set()
            repeated = [x for x in roi.flatten() if x in existed or existed.add(x)]
            total += len(repeated)

    return total


"""
WRITE THIS FUNCTION
"""
def hill_climb(problem, max_sideways=0, max_restarts=0):
    # Given: Sudoku problem and optional max sideways moves and max restarts parameters
    # Return: Board state solution (2d NumPy array), list of errors in each iteration of hill climbing search
    errors_hist = []
    sideways = 0
    restarts = 0
    current = initialize(problem)
    errors_hist.append(num_errors(current))
    while True:
        neighbors = successors(current, problem)
        if len(neighbors) == 0:
            return current, errors_hist
        errors = [num_errors(x) for x in neighbors]
        neighbor = neighbors[errors.index(min(errors))]
        if num_errors(current) == 0:
            return current, errors_hist
        if num_errors(neighbor) > num_errors(current):
            current = initialize(problem)
            restarts += 1

        elif num_errors(neighbor) == num_errors(current) and sideways < max_sideways:
            sideways += 1
            current = neighbor
        elif num_errors(neighbor) == num_errors(current) and sideways >= max_sideways and restarts < max_restarts:
            current = initialize(problem)
            restarts += 1
        elif restarts >= max_restarts:
            return current, errors_hist
        else:
            current = neighbor

        errors_hist.append(num_errors(current))


if __name__ == '__main__':
    # n = 3
    # clues = 40
    # max_sideways = 300
    # max_restarts = 30
    # problem = generate(n, clues)
    # display(problem)
    # sol, errors = hill_climb(problem, max_sideways, max_restarts)
    # print("Solution:\n", sol)
    # print('Last error: ', errors[-1])
    # plt.plot(errors)
    # plt.show()

    #
    n = 2
    clues = 5
    max_sideways = 10
    max_restarts = 10
    batch = 100
    final_errors = []
    success_count = 0
    for i in range(batch):
        problem = generate(n, clues)
        sol, errors = hill_climb(problem, max_sideways, max_restarts)
        final_errors.append(errors[-1])
        if errors[-1] == 0:
            success_count += 1

    avg_success_rate = success_count/len(final_errors)
    avg_final_error = sum(final_errors) / len(final_errors)
    print('Average success rate is: ', avg_success_rate)
    print('Average error over all final states is: ', avg_final_error)
