#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"

    Description:
    The idea of algorithm is Deutsch Jozsa algorithm inside of Deutsch Jozsa.
    On [0, 1, 5] qubits performed classical algorithm with the oracle that combination of four function.
    [0, 1] bits have unitary distribution of states 00, 01, 10, 11 and internal function are controlled by th values.
    Internal function, oracles are performed on [2, 3, 4] bits with result of this function taken to 5th bit
    (if function is ballanced Toffoli gate will change 5th bit) catching "-" for this combination at bits [0, 1].
    After that bits [2, 3, 4] should be returned to the original state by performing same function again.


    Results:
    If state "00" -> all function were either constant (state on [0, 1] bits never changed)
    or balanced (every combination 00, 01, 10, 11 picked up "-" sign -> after applying Hadamard,
    state "00" with global "-"). With multicontroll 5th bit will flip to state |0>

    If state not "00", i.e. two of the combiantion will pick up "-" sign and two not (2 balanced and 2 constant)
    Multicontroll gate never works, bit 5 with state |1>
    """

    # QHACK #
    dev = qml.device("default.qubit", wires = 6)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(4)
        qml.PauliX(5)
        qml.broadcast(qml.Hadamard, range(6), pattern="single")

        for i in range(4):
            ii = tuple(map(int, f'{i:02b}')) #binary representation of numbers (0, 3)
            # that used as control values for functions
            qml.ctrl(fs[i], (0, 1), ii)([2, 3, 4])
            qml.broadcast(qml.Hadamard, [2, 3], pattern="single")
            qml.Toffoli([2, 3, 5])
            #apply function to its own state combination and change bit 5 if function is balanced

            qml.broadcast(qml.Hadamard, [2, 3], pattern="single")
            qml.ctrl(fs[i], (0, 1), ii)([2, 3, 4])
            #apply same function again to bits to return to its orginal state

        qml.broadcast(qml.Hadamard, [0, 1, 5], pattern="single")
        qml.MultiControlledX([0, 1], 5, "00")
        # performing calssical deutsch jozsa and check bits [0, 1]
        # bit 5 is auxilary and dosnt change after algorithm its in state |1>



        return qml.probs([5])

    # print(qml.draw(circuit)()) #draw circuit
    if circuit()[0]:
        return "4 same"
    return "2 and 2"

    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
