from mindquantum import *


circ = Circuit()
n_qubits = 2
circ += UN(H, n_qubits)


circ += BarrierGate(True)

circ += X.on(1)
circ += X.on(0)
circ += PhaseShift(1.5708).on(1,0)
circ += X.on(0)
circ += X.on(1)

circ += BarrierGate(True)
circ += X.on(1)
circ += PhaseShift(1.5708).on(1,0)
circ += X.on(1)



circ += BarrierGate(True)
circ += UN(RY(1.5708), n_qubits)
circ += PhaseShift(1.5708).on(1,0)
circ += UN(RY(-1.5708), n_qubits)
circ += BarrierGate(True)







circ += Measure().on(0)      
circ += Measure().on(1)

sim = Simulator("mqvector", circ.n_qubits)  
res = sim.sampling(circ, shots=10000)


print(circ)
circ.svg()