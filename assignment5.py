import sympy
import numpy as np

Ybus = np.array([
    [-17.361j, 0, 0, 17.361j, 0, 0, 0, 0, 0],
    [0, -16j, 0, 0, 0, 0, 16j, 0, 0],
    [0, 0, -17.065j, 0, 0, 0, 0, 0, 17.065j],
    [17.361j, 0, 0, 3.307-39.309j, -1.365+11.604j, -1.942+10.511j, 0, 0, 0],
    [0, 0, 0, -1.365+11.604j, 2.553-17.338j, 0, -1.188+5.975j, 0, 0],
    [0, 0, 0, -1.942+10.511j, 0, 3.224-15.841j, 0, 0, -1.282+5.588j],
    [0, 16j, 0, 0, -1.188+5.975j, 0, 2.805-35.446j, -1.617+13.698j, 0],
    [0, 0, 0, 0, 0, 0, -1.617+13.698j, 2.772-23.303j, -1.155+9.784j],
    [0, 0, 17.065j, 0, 0, -1.282+5.588j, 0, -1.155+9.784j, 2.437-32.154j]
], dtype=complex)

H = [23.64, 6.4, 3.01] 
Xd = [0.146, 0.8958, 1.3125]  
Xprimed = [0.0608, 0.1198, 0.1813]  
Xq = [0.0969, 0.8645, 1.2578]  
Xprimeq = [0.0969, 0.1969, 0.25]  
Tprimedo = [8.96, 6.0, 5.89]  
Tprimeqo = [0.31, 0.535, 0.6]

Ka = [20, 20, 20]
Ta = [0.2, 0.2, 0.2]
Ke = [1.0, 1.0, 1.0]
Te = [0.314, 0.314, 0.314]
Kf = [0.063, 0.063, 0.063]
Tf = [0.35, 0.35, 0.35]

DoverM = [0.1, 0.2, 0.3]

def A(i):
	return sympy.Matrix([[-1/Tprimedo[i], 0, 0, 0, 1/Tprimedo[i], 0, 0],
				  [0, -1/Tprimeqo[i], 0, 0, 0, 0, 0],
				  [0, 0, 0, 1, 0, 0, 0],
				  [0, 0, 0, -DoverM[i], 0, 0, 0],
				  [0, 0, 0, 0, -Ke[i]/Te[i], 0, 1/Te[i]],
				  [0, 0, 0, 0, Kf[i]/Tf[i]**2, -1/Tf[i], 0],
				  [0, 0, 0, 0, -Ka[i] * Kf[i] / (Ta[i] * Tf[i]), Ka[i] / Ta[i], -1/Ta[i]]])

def R(i):
	return sympy.Matrix([-(Xd[i] - Xprimed[i]) * Id[i] / Tprimedo[i],
				  (Xq[i] - Xprimeq[i]) * Iq[i] / Tprimeqo[i],
				  0,
				  -omega_s/(2*H[i]) * ((Ed[i]*Id[i] + Eq[i]*Iq[i]) + (Xprimeq[i] - Xprimed[i]) * Id[i] * Iq[i]),
				  -SEi(Efd[i]) / Te[i],
				  0,
				  -Ka[i] * V[i]/Ta[i]])

def C(i):
	return sympy.Matrix([[0, 0, 0],
				  [0, 0, 0],
				  [-1, 0, 0],
				  [DoverM[i], omega_s/(2*H[i]), 0],
				  [0, 0, 0],
				  [0, 0, 0],
				  [0, 0, Ka[i]/Ta[i]]])



# Create symbols Eqi, Edi, deltai, omegai, Efdi, Rfi, and Vri for i = 1, 2, 3
Eq = sympy.symbols('Eq1 Eq2 Eq3')
Ed = sympy.symbols('Ed1 Ed2 Ed3')
delta = sympy.symbols('delta1 delta2 delta3')
omega = sympy.symbols('omega1 omega2 omega3')
Efd = sympy.symbols('Efd1 Efd2 Efd3')
Rf = sympy.symbols('Rf1 Rf2 Rf3')
Vr = sympy.symbols('Vr1 Vr2 Vr3')

Tm = sympy.symbols('Tm1 Tm2 Tm3')
omega_s = sympy.symbols('omega_s')
Vref = sympy.symbols('Vref1 Vref2 Vref3')


theta = sympy.symbols('theta1 theta2 theta3 theta4 theta5 theta6 theta7 theta8 theta9')
V = sympy.symbols('V1 V2 V3 V4 V5 V6 V7 V8 V9')

def u(i):
	return sympy.Matrix([omega_s, Tm[i], Vref[i]])

def SEi(Efd):
	return 0.0039 * sympy.exp(1.555 * Efd)

def Iqf(i):
	return (V[i] * sympy.sin(delta[i] - theta[i]) - Ed[i]) / Xprimeq[i]

def Idf(i):
	return (Eq[i] - V[i] * sympy.cos(delta[i] - theta[i])) / Xprimed[i]


Iq1 = Iqf(0)
Iq2 = Iqf(1)
Iq3 = Iqf(2)

Iq = [Iq1, Iq2, Iq3]

Id1 = Idf(0)
Id2 = Idf(1)
Id3 = Idf(2)

Id = [Id1, Id2, Id3]

# Active Power equations
active1 = Id[0] * V[0] * sympy.sin(delta[0] - theta[0]) + Iq[0] * V[0] * sympy.cos(delta[0] - theta[0]) - (17.361 * V[0] * V[3] * sympy.sin(theta[0] - theta[3]))
active2 = Id[1] * V[1] * sympy.sin(delta[1] - theta[1]) + Iq[1] * V[1] * sympy.cos(delta[1] - theta[1]) - (16.0 * V[1] * V[6] * sympy.sin(theta[1] - theta[6]))
active3 = Id[2] * V[2] * sympy.sin(delta[2] - theta[2]) + Iq[2] * V[2] * sympy.cos(delta[2] - theta[2]) - (17.065 * V[2] * V[8] * sympy.sin(theta[2] - theta[8]))

# Reactive power equations
reactive1 = Id[0] * V[0] * sympy.cos(delta[0] - theta[0]) - Iq[0] * V[0] * sympy.sin(delta[0] - theta[0]) - (17.361 * V[0]**2 - 17.361 * V[0] * V[3] * sympy.cos(theta[0] - theta[3]))
reactive2 = Id[1] * V[1] * sympy.cos(delta[1] - theta[1]) - Iq[1] * V[1] * sympy.sin(delta[1] - theta[1]) - (16.0 * V[1]**2 - 16.0 * V[1] * V[6] * sympy.cos(theta[1] - theta[6]))
reactive3 = Id[2] * V[2] * sympy.cos(delta[2] - theta[2]) - Iq[2] * V[2] * sympy.sin(delta[2] - theta[2]) - (17.065 * V[2]**2 - 17.065 * V[2] * V[8] * sympy.cos(theta[2] - theta[8]))

print(active1)
print(active2)
print(active3)
print(reactive1)
print(reactive2)
print(reactive3)
print("-----------------")


def dot(i):
	return A(i) @ sympy.Matrix([Eq[i], Ed[i], delta[i], omega[i], Efd[i], Rf[i], Vr[i]]) + R(i) + C(i) @ u(i)

print(dot(0))
print(dot(1))
print(dot(2))

