import sys
import pennylane as qml
from pennylane import numpy as np

NUM_WIRES = 6


def triple_excitation_matrix(gamma):
    """The matrix representation of a triple-excitation Givens rotation.

    Args:
        - gamma (float): The angle of rotation

    Returns:
        - (np.ndarray): The matrix representation of a triple-excitation
    """

    # QHACK #
    U = np.eye(64)
    U[7,7] = np.cos(gamma/2)
    U[56,56] = np.cos(gamma/2)
    U[7,56] = np.sin(gamma/2)
    U[56,7] = -np.sin(gamma/2)
    return U

    # QHACK #


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev)
def circuit(angles):
    """Prepares the quantum state in the problem statement and returns qml.probs

    Args:
        - angles (list(float)): The relevant angles in the problem statement in this order:
        [alpha, beta, gamma]

    Returns:
        - (np.tensor): The probability of each computational basis state
    """

    # QHACK #
    qml.PauliX(0)
    qml.PauliX(1)
    qml.PauliX(2)
    qml.SingleExcitation(angles[0],[0,5])
    qml.DoubleExcitation(angles[1],[0,1,4,5])
    qml.QubitUnitary(triple_excitation_matrix(angles[2]),wires=range(6))

    # QHACK #

    return qml.probs(wires=range(NUM_WIRES))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = np.array(sys.stdin.read().split(","), dtype=float)
    probs = circuit(inputs).round(6)
    print(*probs, sep=",")
