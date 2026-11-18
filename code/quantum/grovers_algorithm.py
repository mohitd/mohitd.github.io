import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler

A, B, C = 0, 1, 2
# clause qubits
C1, C2, C3, C4 = 3, 4, 5, 6
OUT = 7

QUBIT_NAMES = ["a", "b", "c", "a ∨ b", "~b ∨ c", "~a ∨ c", "~b ∨ ~c", "out"]
QREGS = [QuantumRegister(1, name) for name in QUBIT_NAMES]
N_VARS = 3


def new_circuit(name: str, measure: bool = False) -> QuantumCircuit:
    cregs = [ClassicalRegister(N_VARS, "meas")] if measure else []
    return QuantumCircuit(*QREGS, *cregs, name=name)


def f() -> QuantumCircuit:
    qc = new_circuit("U_f")

    # C1 = a or b = not (not a and not b)
    qc.x([A, B])
    qc.ccx(A, B, C1)
    qc.x(C1)
    # remember to undo A, B!
    qc.x([A, B])

    # C2 = not b or c = not (b and not c)
    qc.x(C)
    qc.ccx(B, C, C2)
    qc.x(C2)
    # remember to undo C!
    qc.x(C)

    # C3 = not a or c = not (a and not c)
    qc.x(C)
    qc.ccx(A, C, C3)
    qc.x(C3)
    # remember to undo C!
    qc.x(C)

    # C4 = not b or not c = not (b and c)
    qc.ccx(B, C, C4)
    qc.x(C4)

    return qc


def oracle() -> QuantumCircuit:
    f_x = f()

    qc = new_circuit("oracle")
    # compute f
    qc.compose(f_x, inplace=True)
    qc.barrier(label="compute")

    # AND the clauses together into the output ancilla
    qc.mcx([C1, C2, C3, C4], OUT)
    qc.barrier(label="kickback")

    # remember to uncompute the clauses!
    qc.compose(f_x.inverse(), inplace=True)
    qc.barrier(label="uncompute")
    return qc


def diffusion() -> QuantumCircuit:
    qc = new_circuit("diffusion")
    qc.h([A, B, C])
    qc.x([A, B, C])

    qc.ccz(A, B, C)

    qc.x([A, B, C])
    qc.h([A, B, C])
    return qc


def grover(k: int) -> QuantumCircuit:
    qc = new_circuit("grovers", measure=True)

    # even superposition over the data qubits
    qc.h([A, B, C])
    # remember to put the output ancilla in |-> = HX|0>.
    qc.x(OUT)
    qc.h(OUT)
    qc.barrier()

    # run the oracle+diffusion combination k times
    for _ in range(k):
        qc.compose(oracle(), wrap=True, inplace=True)
        qc.compose(diffusion(), wrap=True, inplace=True)
        qc.barrier()

    # measure a, b, c into classical registers
    qc.measure([A, B, C], range(N_VARS))
    return qc


def is_solution(abc: str) -> bool:
    # verifying a candidate is cheap: just evaluate f classically
    a, b, c = (bit == "1" for bit in abc)
    return (a or b) and (not b or c) and (not a or c) and (not b or not c)


def main() -> None:
    # cheating a little and we know there is only 1 solution to this problem
    num_states, num_solutions = 2**N_VARS, 1
    theta = math.asin(math.sqrt(num_solutions / num_states))
    k = round(math.pi / (4 * theta) - 0.5)
    print(f"theta = {theta:.3f}, iterations k = {k}")
    print(f"predicted success = sin^2((2k + 1) theta) = {math.sin((2 * k + 1) * theta) ** 2:.3f}")

    # Draw each circuit
    print("\nU_f")
    print(f().draw(output="text"))
    print("\nOracle")
    print(oracle().draw(output="text"))
    print("\nDiffusion")
    print(diffusion().draw(output="text", idle_wires=False))

    qc = grover(k)
    print(f"\nFull circuit with k = {k}")
    print(qc.draw(output="text"))
    print("\n")

    shots = 2 ** 10
    result = StatevectorSampler(seed=42).run([qc], shots=shots).result()
    counts = result[0].data.meas.get_counts()
    print(f"measured counts (abc, {shots} shots):")
    for bits, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        # reverse the order for readability since Qiskit uses reverse bit order
        print(f"  {bits[::-1]}: {count}")
    successes = sum(count for bits, count in counts.items() if is_solution(bits[::-1]))
    print(f"measured success = {successes / shots:.3f}")


if __name__ == "__main__":
    main()
