#!/usr/bin/env python
# coding: utf-8

# In[4]:


from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st




def qubit2_tomography(N):

    N=N/(N[0] + N[1] + N[4] + N[5])
    
    basis = np.array([np.array([1, 0]), np.array([0, 1]), np.array([1/np.sqrt(2), 1/np.sqrt(2)]), np.array([1/np.sqrt(2), 1j/np.sqrt(2)])])
    basis_vecs = []
    for i in range(4):
        for j in range(4):
            basis_vecs.append(np.outer(np.kron(basis[i],basis[j]),np.conj(np.kron(basis[i],basis[j]))))
    E=np.array([0.998,1.0146,0.9195,0.9265])
    Evec = []
    for i in range(4):
        for j in range(4):
            Evec.append(E[i]*E[j])


    def objective_func(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]
        x7 = x[6]
        x8 = x[7]
        x9 = x[8]
        x10 = x[9]
        x11 = x[10]
        x12 = x[11]
        x13= x[12]
        x14= x[13]
        x15= x[14]
        x16= x[15]


        T_d = np.array([[x[0], 0, 0, 0], [x[4]+ 1j * x[5], x[1], 0, 0], [x[10] + 1j * x[11], x[6] + 1j * x[7], x[2], 0], [x[14] + 1j * x[15], x[12] + 1j * x[13], x[8] + 1j * x[9], x[3]]])
        rho=np.conj(T_d).T @ T_d
        L=0
        for i in range(16):

            L += (Evec[i]*(np.trace(basis_vecs[i] @ rho)) -N[i]+(19/3178))**2 / (2 * np.trace(basis_vecs[i] @ rho))
#             L += ((np.trace(basis_vecs[i] @ rho)) -N[i])**2 / (2 * np.trace(basis_vecs[i] @ rho))
        return np.real(L)


    def cons2(x):
        u=0
        for i in range (0,16):
            u=u+x[i]**2
        return u-1
#         return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 1+ x[4]**2+ x[5]**2+ x[6]**2+ x[7]**2+ x[8]**2+ x[9]**2+ x[10]**2+ x[11]**2+ x[12]**2+ x[13]**2+ x[14]**2+ x[15]**2

    
    x0=np.array([1 for x0 in range(16)])


    #def min_obj_func2(N, x0):
    cons = [{'type': 'eq', 'fun': cons2}]
    solution = minimize(objective_func, x0, method='SLSQP', constraints=cons)


    x=solution.x
    
    T_d = np.array([[x[0], 0, 0, 0], [x[4]+ 1j * x[5], x[1], 0, 0], [x[10] + 1j * x[11], x[6] + 1j * x[7], x[2], 0], [x[14] + 1j * x[15], x[12] + 1j * x[13], x[8] + 1j * x[9], x[3]]])
    rho=np.conj(T_d).T @ T_d



   
#     print(is_density_matrix(rho))
#     print(concurrence(rho))

    fig = plt.figure()
    ax = fig.add_subplot(122, projection='3d')
    ax1= fig.add_subplot(121, projection='3d')
    x_data = np.array([0,1 , 2, 3])
    y_data = np.array([0, 1, 2,3])
    z_data = np.imag(rho)
    z_data2=np.real(rho)
    dx = dy = 0.5  # width of each bar in x and y direction
    dz = z_data.ravel()  # height of each bar
    dz1=z_data2.ravel()
    x, y = np.meshgrid(x_data, y_data)
    x, y, z = x.ravel(), y.ravel(), 0

    # Plot 3D bars
    ax.bar3d(x, y, z, dx, dy, dz)
    ax1.bar3d(x, y, z, dx, dy, dz1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_zlim(-1,1)


    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Amplitude')
    ax1.set_zlim(-1,1)
    plt.show()
    
    
    return np.round(rho,4),fig


st.title("Two Qubit Tomography")

a = st.text_input("HH")
b = st.text_input("HV")
c = st.text_input("HD")
d = st.text_input("HL")
e = st.text_input("VH")
f = st.text_input("VV")
g = st.text_input("VD")
h = st.text_input("VL")
j = st.text_input("DH")
k = st.text_input("DV")
l = st.text_input("DD")
m = st.text_input("DL")
n = st.text_input("LH")
o = st.text_input("LV")
p = st.text_input("LD")
q = st.text_input("LL")

if a != '' and b != '' and c != '' and d != '' and e != '' and f != '' and g != '' and h != '' and j != '' and k != '' and l != '' and m != '' and n != '' and o != '' and p != '' and q != '':

	N_all_0 = [a, b, c, d, e, f, g, h, j, k, l, m, n, o, p, q]
	N_all = []
	for x in N_all_0:
		N_all.append(float(x))
	
	N_all=np.array(N_all)
	rho = qubit2_tomography(N_all)

	st.write(rho[0])
	st.pyplot(rho[1])
	

# In[ ]:




