#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    N = np.linalg.norm([alpha, beta])
    U = np.array([[alpha/N, beta/N], [beta/N, - alpha/N]])
    qml.QubitUnitary(U, 0)
    qml.CNOT([0, 1])
    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    if x:
        theta_a = theta_A1
    else:
        theta_a = theta_A0
    if y:
        theta_b = theta_B1
    else:
        theta_b = theta_B0
    U_a = np.array([[np.cos(theta_a), np.sin(theta_a)], [-np.sin(theta_a), np.cos(theta_a)]])
    U_b = np.array([[np.cos(theta_b), np.sin(theta_b)], [-np.sin(theta_b), np.cos(theta_b)]])

    qml.QubitUnitary(U_a, 0)
    qml.QubitUnitary(U_b, 1)

    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    # QHACK #
    prob = 0
    for x in [0, 1]:
        for y in [0, 1]:
            res = chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)
            prob += 1/4 * (x*y*(res[1] + res[2]) + (1 - x*y)*(res[0] + res[3]))
    return prob

    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""

    # QHACK #
        return -winning_prob(params, alpha, beta)

    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([0.01, 0.01, 0.01, 0.01], requires_grad=True)
    opt = qml.AdamOptimizer()
    steps = 100

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #

        params = opt.step(cost, params)
        params = np.clip(opt.step(cost, params), -2*np.pi, 2*np.pi)

        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")