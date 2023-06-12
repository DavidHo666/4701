from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

"""
WRITE THIS FUNCTION
"""
def value_iteration(
    V0: npt.NDArray,
    lr: float,
    gamma:float,
    epsilon: float=1e-12
    ) -> npt.NDArray:
    diff = [float('inf')] * 22
    while any(x > epsilon for x in diff):
        for s in range(22):
            next_val = 0
            for card in range(1, 11, 1):
                if s + card > 21:
                    break
                else:
                    if card != 10:
                        next_val += 1 / 13 * (lr + gamma * V0[s + card])
                    else:
                        next_val += 4 / 13 * (lr + gamma * V0[s + card])
            v_i_plus1 = max(next_val, s)
            diff[s] = abs(v_i_plus1 - V0[s])
            V0[s] = v_i_plus1

    return V0


def value_to_policy(V: npt.NDArray, lr: float, gamma: float) -> npt.NDArray:
    actions = np.zeros(V.size)
    for s in range(22):
        next_val = 0
        for card in range(1,11,1):
            if s+card > 21:
                break
            else:
                if card != 10:
                    next_val += 1/13*(lr+gamma*V[s+card])
                else:
                    next_val += 4/13*(lr+gamma*V[s+card])

        actions[s] = 1 if next_val>s else 0
    return actions


def draw() -> int:
    probs = 1 / 13 * np.ones(10)
    probs[-1] *= 4
    return np.random.choice(np.arange(1, 11), p=probs)


"""
WRITE THIS FUNCTION
"""


def Qlearn(
        Q0: npt.NDArray,
        lr: float,
        gamma: float,
        alpha: float,
        epsilon: float,
        N: int
) -> Tuple[npt.NDArray, npt.NDArray]:
    import random
    record = np.zeros((N,3))
    s = 0
    s_prime = None
    reward = None
    for i in range(N):
        if Q0[s][0]==0 and Q0[s][1] == 0:
            a = 1 if random.random()>0.5 else 0
        else:
            a = np.argmax(Q0[s]) if random.random()<(1-epsilon+epsilon/2) else np.argmin(Q0[s])
        if a == 0:
            Q0[s][a] = Q0[s][a] + alpha*(s - Q0[s][a])
            s_prime = 0
            reward = s
        elif a == 1:
            card = draw()
            if s+card >21:
                Q0[s][a] = Q0[s][a] + alpha*(0 - Q0[s][a])
                s_prime = 0
                reward = 0
            else:
                Q0[s][a] = Q0[s][a] + alpha*(lr + gamma*np.max(Q0[s+card]) - Q0[s][a])
                s_prime = s + card
                reward = lr

        record[i][0]=s
        record[i][1]=a
        record[i][2]= reward
        s = s_prime

    return Q0, record