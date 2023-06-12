# dh3027 Dawei He
import enchant, string
import matplotlib.pyplot as plt
from heapq import heappush, heappop

def successors(state):
    """
    Given a word, find all possible English word results from changing one letter.
    Return a list of (action, word) pairs, where action is the index of the
    changed letter.
    """
    d = enchant.Dict("en_US")
    child_states = []
    for i in range(len(state)):
        new = [state[:i]+x+state[i+1:] for x in string.ascii_lowercase]
        words = [x for x in new if d.check(x) and x != state]
        child_states = child_states + [(i, word) for word in words]
    return child_states


"""
5.1: Best-first search
"""
def best_first_search(start, goal, f):
    """
    Inputs: Start state, goal state, priority function
    Returns node containing goal or None if no goal found, total nodes expanded,
    frontier size per iteration
    """
    node = {'state':start, 'parent':None, 'cost':0}
    frontier = []
    reached = {}
    nodes_expanded = 0
    frontier_size = [len(frontier)]


    # COMPLETE THIS FUNCTION
    heappush(frontier, tuple([f(node, goal)]+[ch for ch in node['state']]+[node]))
    reached[node['state']] = node['cost']

    while frontier:
        frontier_size.append(len(frontier))
        node = heappop(frontier)[-1]
        nodes_expanded += 1
        if node['state'] == goal:
            return node, nodes_expanded, frontier_size
        words = successors(node['state'])
        for idx, word in words:
            if word not in reached or node['cost']+1 < reached[word]:
                reached[word] = node['cost']+1
                new_node = {'state':word,
                            'parent':node,
                            'cost':node['cost']+1}
                heappush(frontier, tuple([f(new_node, goal)]+
                                         [x for x in new_node['state']]+
                                         [new_node]))

    return None, nodes_expanded, frontier_size


"""
5.2: Priority functions
"""
def f_dfs(node, goal=None):
    # IMPLEMENT THIS FUNCTION
    return -node['cost']

def f_bfs(node, goal=None):
    # IMPLEMENT THIS FUNCTION
    return node['cost']

def f_ucs(node, goal=None):
    # IMPLEMENT THIS FUNCTION
    return node['cost']

def f_astar(node, goal):
    # IMPLEMENT THIS FUNCTION
    h = 0
    goal_list = [ch for ch in goal]
    word_list = [ch for ch in node['state']]
    for i in range(len(goal)):
        if goal_list[i]!=word_list[i]:
            h += 1

    return node['cost'] + h


def sequence(node):
    """
    Given a node, follow its parents back to the start state.
    Return sequence of words from start to goal.
    """
    words = [node['state']]
    while node['parent'] is not None:
        node = node['parent']
        words.insert(0, node['state'])
    return words


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    start = 'small'
    goal = 'large'
    print(f'Start: {start}\nGoal: {goal}\n')

    # algo = 'DFS'
    # begin = time.time()
    # solution = best_first_search(start, goal, f_dfs)
    # end = time.time()
    # path = sequence(solution[0])
    # print(f'The solution path of {algo} algorithm is:\n{path}')
    # print(f'The {algo} algorithm execution time is: {end-begin}')
    # path = sequence(solution[0])
    # print(f'The length of the solution of {algo} algorithm is: {len(path)}')
    # print(f'The number of nodes expanded of {algo} algorithm is: {solution[1]}')
    # print()

    algo = 'BFS'
    begin = time.time()
    solution = best_first_search(start, goal, f_bfs)
    end = time.time()
    path = sequence(solution[0])
    print(f'The solution path of {algo} algorithm is:\n{path}')
    bfs_frontier_size = solution[2]
    print(f'The {algo} algorithm execution time is: {end-begin}')
    path = sequence(solution[0])
    print(f'The length of the solution of {algo} algorithm is: {len(path)}')
    print(f'The number of nodes expanded of {algo} algorithm is: {solution[1]}')
    print()

    algo = 'UCS'
    begin = time.time()
    solution = best_first_search(start, goal, f_ucs)
    end = time.time()
    path = sequence(solution[0])
    print(f'The solution path of {algo} algorithm is:\n{path}')
    ucs_frontier_size = solution[2]
    print(f'The {algo} algorithm execution time is: {end-begin}')
    path = sequence(solution[0])
    print(f'The length of the solution of {algo} algorithm is: {len(path)}')
    print(f'The number of nodes expanded of {algo} algorithm is: {solution[1]}')
    print()

    algo = 'A*'
    begin = time.time()
    solution = best_first_search(start, goal, f_astar)
    end = time.time()
    path = sequence(solution[0])
    print(f'The solution path of {algo} algorithm is:\n{path}')
    astar_frontier_size = solution[2]
    print(f'The {algo} algorithm execution time is: {end-begin}')
    path = sequence(solution[0])
    print(f'The length of the solution of {algo} algorithm is: {len(path)}')
    print(f'The number of nodes expanded of {algo} algorithm is: {solution[1]}')

    plt.plot(bfs_frontier_size, label = 'BFS')
    plt.plot(ucs_frontier_size, label = 'UCS')
    plt.plot(astar_frontier_size, label = 'A*')
    plt.xlabel('The number of iterations')
    plt.ylabel('Frontier size')
    plt.legend()
    plt.title(f'Start: {start}, Goal: {goal}')
    plt.show()

    # print(solution[0])
