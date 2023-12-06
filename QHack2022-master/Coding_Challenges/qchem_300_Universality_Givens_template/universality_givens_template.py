#! /usr/bin/python3

import sys
import numpy as np


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #
    import pennylane as qml
    from pennylane import numpy as np
    dev = qml.device("default.qubit", wires=6)
    @qml.qnode(dev)

    def circuit(theta_1, theta_2, theta_3):
        qml.PauliX(0)
        qml.PauliX(1)
        qml.DoubleExcitation(theta_1, range(4))
        qml.DoubleExcitation(theta_2, [2, 3, 4, 5])
        qml.ctrl(qml.SingleExcitation, 0)(theta_3, [1, 3])
        return qml.state()


    def cost(params):
        res = np.real(circuit(params[0], params[1], params[2]))
        a_res = res[48]
        b_res = res[14]
        c_res = res[3]
        d_res = res[36]
        return np.sqrt((a - a_res)**2 + (b - b_res)**2 + (c - c_res)**2 + (d - d_res)**2)


    init_params = np.array([0.001, 0.001, 0.001], requires_grad=True)
    opt = qml.AdamOptimizer()
    steps = 500

    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)
        params = np.clip(opt.step(cost, params), -2 * np.pi, 2 * np.pi)

    return params
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
