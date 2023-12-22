import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(
    page_title="Quantifyhub",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for light mode
light_mode_css = """
<style>
body {
    color: black;
    background-color: white;
}
</style>
"""
st.markdown(light_mode_css, unsafe_allow_html=True)


def introduction_page():
    # Header
    st.title("QuantifyHub")
    st.write("Unveiling Quantum Frontiers")

    # About Us
    st.header("About Us")
    st.write(
        "QuantifyHub is at the forefront of quantum research, driven by a team of passionate physicists, software developers, and researchers."
        " Explore the quantum realm with our innovative software solutions."
    )

    # Mission and Vision
    #st.header("Mission and Vision")
    st.write(" Accelerate quantum research through software tools and resources.")
    st.write(" Assist individuals globally by providing software tools and guidance for developing SPDC sources, contributing to advancements in quantum research and technology.")


    # Key Features
    st.header("Key Features")
    features = {
        "Quantum State Tomography": "Advanced toolkit for measuring and reconstructing quantum states.",
        "SPDC Sources": "Simulates Spontaneous Parametric Down-Conversion for quantum optics experiments.",
        "Phase Matching Angle BBO": "Calculates phase matching angles for nonlinear crystals like Beta-Barium Borate.",
        "Compensation Crystal Length": "Optimizes crystal lengths for compensating phase mismatch in quantum processes.",
        "PPKTP QPM Type(0 and 2)": "Addresses quantum phase matching in Periodically Poled Potassium Titanyl Phosphate."
    }

    for feature, description in features.items():
        st.write(f"- **{feature}:** {description}")

    # Call-to-Action Button
    st.button("Explore Our Quantum Software")

    # Contact Information
    st.header("Contact Us")
    st.write("For inquiries or collaboration opportunities, reach out to us at umesh3210joshi@gmail.com")



def software_list_page():
    st.title("Quantum State Tomography Interface")
    st.write("")
    st.title("Enter 16 Measuremet Values")
    default_value = 1024
    value_names = ["HH", "HV", "HD", "HL", "VH", "VV", "VD", "VL", "DH", "DV", "DD", "DL", "LH", "LV", "LD", "LL"]
    cols = st.columns(4)
    values = {}
    for i, name in enumerate(value_names):
        col = cols[i % 4]
        value = col.number_input(f"{name}", min_value=0, max_value=1000000000, value=default_value, step=1)
        values[name] = value

        
    st.write("Entered Values:", values)
    values_array = np.array(list(values.values()))
    
    rho, fig3 = qubit2_tomography(values_array)
    st.header("Results")
    st.write(f"Density Matrix: {rho} ")

    phi_plus = 1/(2) * np.array([[1, 0, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [1, 0, 0, 1]])

    phi_minus = 1/(2) * np.array([[1, 0, 0, -1],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [-1, 0, 0, 1]])

    psi_plus = 1/(2) * np.array([[0, 0, 0, 0],
                                 [0, 1, 1, 0],
                                 [0, 1, 1, 0],
                                 [0, 0, 0, 0]])

    psi_minus = 1/(2) * np.array([[0, 0, 0, 0],
                                  [0, 1, -1, 0],
                                  [0, -1, 1, 0],
                                  [0, 0, 0, 0]])

    for i, bell_state in enumerate([phi_plus, phi_minus, psi_plus, psi_minus], start=1):
        fid = fidelity(rho, bell_state)
        st.warning(f"Fidelity with Bell State {i}: {fid:.15f}")
    concurrence_value = concurrence(rho)
    st.error(f"Concurrence: {concurrence_value:.15f}")
    negativity_value = negativity(rho)
    st.warning(f"Negativity: {negativity_value}")

    linear_entropy_value = linear_entropy(rho)
    st.error(f"Linear Entropy: {linear_entropy_value:.15f}")

    entropy_value = entropy(rho)
    st.info(f"Von Neumann Entropy: {entropy_value:.15f}")

    # Calculate and print purity
    purity_value = purity(rho)
    st.error(f"Purity: {purity_value:.15f}")


    st.pyplot(fig3)

import streamlit as st
import numpy as np
from scipy.linalg import sqrtm

def fidelity(rho, expected_state):
    sqrt_rho = sqrtm(rho)
    sqrt_expected_state = sqrtm(expected_state)
    inner_product = np.trace(np.matmul(np.matmul(sqrt_rho, sqrt_expected_state), sqrt_rho))
    fid = np.abs(inner_product) ** 2
    return fid



def purity(rho):
    # Calculate the purity of a quantum state
    trace_rho_squared = np.trace(np.dot(rho, rho))
    purity_value = np.real(trace_rho_squared)
    
    return purity_value

def negativity(rho):
    # Calculate the negativity for a two-qubit state
    rho_B = np.trace(np.reshape(rho, (2, 2, 2, 2)), axis1=0, axis2=2)
    neg = 0.5 * (np.linalg.norm(ppt(rho,2,1)) - 1)
    
    return max(neg, 0)

def linear_entropy(rho):
    # Calculate the linear entropy for a quantum state
    linear_ent = 1 - np.trace(np.dot(rho, rho))
    
    return max(linear_ent, 0)

def entropy(rho):
    # Calculate the von Neumann entropy for a quantum state
    eigvals = np.linalg.eigvalsh(rho)
    entropy_val = -np.sum(np.nan_to_num(eigvals * np.log2(eigvals)))
    
    return max(entropy_val, 0)



def phase_matching_type12():
    st.title("Phase Matching Angle")
    wavelength = st.slider("Select Wavelength (nm)", min_value=300, max_value=1000, 	value=405)
    angle1 ,angle2 ,fig = phase_matching_type1_and_type2(wavelength)

    # Display results
    st.header("Results")
    st.write(f"Selected Wavelength: {wavelength} nm")
    st.write(f"Phase Matching Angle: {angle1} degrees")
    st.write(f"Phase Matching Angle: {angle2} degrees")
    st.pyplot(fig)
    
def phase_matching_ppktp():
    st.title("Phase Matching Temperature PPKTP")
    wavelength = st.slider("Select Wavelength (nm)", min_value=300, max_value=1000, 	value=405)
    fig = phase_matching_temp(wavelength)
    st.plotly_chart(fig)
    a = st.slider("Select starting Temperature", min_value=20, max_value=100, 	value=27)
    b = st.slider("Select starting Temperature", min_value=20, max_value=100, 	value=50)
    c = st.slider("Select the angle", min_value=2, max_value=90,value=7)
    period = st.number_input("Enter a number", min_value=0.0, max_value=4.0, value=3.425, step=0.001)
    fig = degeneracy_graph_ppktp(a,b,period)
    st.plotly_chart(fig)
    fig = angle_for_degeneracy(wavelength,a,b,c)
    st.plotly_chart(fig)
    
def phase_matching_page():
    st.title("Phase Matching Angle Calculation")
    wavelength = st.slider("Select Wavelength (nm)", min_value=300, max_value=1000, 	value=405)

    # Calculate phase matching angle
    angle, fig = phase_matching_angle_calculation(wavelength)

    # Display results
    st.header("Results")
    st.write(f"Selected Wavelength: {wavelength} nm")
    st.write(f"Phase Matching Angle: {angle:.2f} degrees")
    st.pyplot(fig)
    
def projects():
    image_url = https://drive.google.com/file/d/151eXFuuBMj_8_P1nMdbHoBBpKOQo4BgM/view?usp=sharing
    st.image(image_url, caption='', use_column_width=True)


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

def angle_for_degeneracy(wavelength1,a,b,input_theta):
    mode=1
    fixed_period=3.425
    temperature_values = np.linspace(a, b, num=5)  # Adjust the number of temperatures as needed
    result_values = {}

    for temperature in temperature_values:
        wavelength2 = wavelength1*2
        wavelength3 = (1 / wavelength1 - 1 / wavelength2) ** -1
        n1 = calculate_n_zt(wavelength1, temperature) / (wavelength1 / 1000)

        theta_values_deg = np.linspace(0, input_theta, num=100)  # Adjust the number of theta values as needed
        theta_values_rad = np.radians(theta_values_deg)

        n2_values = []
        for theta_rad in theta_values_rad:
            n2 = ((nxz(theta_rad, wavelength2, temperature)) /
                  (wavelength2 / 1000) +
                  (nxz(theta_rad, wavelength3, temperature)) /
                  (wavelength3 / 1000)) * np.cos(theta_rad) + ((mode) /
                                                               (fixed_period))
            n2_values.append(np.abs(n1 - n2))

        result_values[temperature] = {'theta_deg': theta_values_deg/2, 'n2_values': n2_values}

    # Create a plot for each temperature
    fig = go.Figure()

    for temperature, data in result_values.items():
        fig.add_trace(go.Scatter(
            x=data['theta_deg'],
            y=data['n2_values'],
            mode='lines', 
            name=f'Temperature {temperature}°C'
        ))

    fig.update_layout(
        xaxis=dict(title='Theta (degrees)'),
        yaxis=dict(title='np.abs(n1 - n2)'),
        title='np.abs(n1 - n2) vs Theta at Different Temperatures with Plotly',
        template="plotly_dark" 
    )

#     pyo.iplot(fig)
    return fig
# angle_for_degeneracy(405,22,45)
def phase_matching_type1_and_type2(lamb):
    
    fig= plt.figure(figsize = (10, 4))
    for b in range(0,1):
        if b==0:
            st='BBO'
            # Define the constants
            a1, b1, c1, d1 = 2.7405, 0.0184, 0.0179, 0.0155
            a2, b2, c2, d2 = 2.373, 0.0128, 0.0156, 0.0044
            wavelengths =  [lamb, 2*lamb]

            def calc_no(l):
                l = l / 1000
                return np.sqrt(a1 + (b1) / (l**2 - c1) - d1 * (l**2))
            def calc_ne(l):
                l = l / 1000
                return np.sqrt(a2 + ((b2) / (l**2 - c2)) - d2 * (l**2))

        if b==1:
            st='KDP'
            a1, b1, c1, d1, e1 = 2.259276, 0.01008956, 0.012942625, 13.00522, 400
            a2, b2, c2, d2, e2 = 2.132668, 0.008637494, 0.012281043, 3.2279924, 400
            wavelengths = [lamb, lamb*2]

            def calc_no(l):
                l = l / 1000
                no = a1 + (b1 / (l**2 - c1)) + (d1*(l**2) / (l**2 - e1))
                return no

            def calc_ne(l):
                l = l / 1000
                ne = a2 + (b2 / ((l**2) - c2)) + (d2*(l**2) / ((l**2) - e2))
                return ne

        ne_values = [calc_ne(l) for l in wavelengths]
        no_values = [calc_no(l) for l in wavelengths]

        print()

        def function1(theta):
            s = ((np.cos(theta))**2/no_values[0]**2) + ((np.sin(theta))**2/ne_values[0]**2)
            nee = np.sqrt(1/s)
            k = ((np.cos(theta))**2/no_values[1]**2) + ((np.sin(theta))**2/ne_values[1]**2)
            nex = np.sqrt(1/k)
            func=2*nee- (nex*np.cos((3) * np.pi / 180)+ (no_values[1]*np.cos((3) * np.pi / 180)))
            return np.abs(func)

        def function2(theta):
            s = ((np.cos(theta))**2/no_values[0]**2) + ((np.sin(theta))**2/ne_values[0]**2)
            nee = np.sqrt(1/s)
            func=nee-(no_values[1]*np.cos((3) * np.pi / 180))
            return np.abs(func)


        x0 = 1
        # Call the minimize function
        result = minimize(function1, x0)
        result2 = minimize(function2, x0)

        # Extract the minimum value and the corresponding argument
        min_value = result.fun
        min_argument = result.x * 180 / np.pi
        min_value2 = result2.fun
        min_argument2 = result2.x * 180 / np.pi

        plt.subplot(1, 2, b + 1)
        plt.title(f"{st}")
        theta=np.linspace(0,3.14/2,90)
        plt.plot(function1(theta),label= 'TYPE 2')
        theta=np.linspace(0,3.14/2,90)
        plt.plot(function2(theta),label= 'TYPE 1')
        plt.xlabel('PM theta')
        plt.grid(True, which='both', linestyle=':', color='red', linewidth=1.5, alpha=0.6)

        plt.legend()

        # Print the results
        print(f"Pump and Signal/idler wavelengths: {wavelengths}")
        print(f"n(e ray) values {st}: {ne_values}")
        print(f"n(o ray) values {st}: {no_values}")
        print('')
        print(f"Minimum value: Type 1 {min_value2}")
        print(f"Phase matching angle for {st}{min_argument2}")
        print(f"Minimum value: Type 2 { min_value}")
        print(f"Phase matching angle for {st}:Type 2 {min_argument}")
        
    return min_argument2, min_argument , fig

def phase_matching_angle_calculation(lamb):

    """

    Author: [Decoherence]
    Date: [11 04 2023]

    Description: This code calculates the phase matching angle for Type 1 different nonlinear crystals, BBO (Beta-Barium Borate) and
    KTP (Potassium Titanyl Phosphate). It uses the Sellmeier equations  to calculate 
    the refractive indices for the extraordinary (ne) and ordinary (no) rays at given
    wavelengths, and then finds the angle at which the phase matching condition    is 
    satisfied for each crystal. 
    """

    # Define the constants
    a1, b1, c1, d1,e1 = 2.259276,0.01008956, 0.012942625,13.00522 , 400
    a2, b2, c2, d2,e2 = 2.132668, 0.008637494,0.012281043,3.2279924 , 400

    # Define the functions
    def calc_no(l):
        l = l / 1000
        no =a1 + (b1 / (l**2 - c1)) + (d1*(l**2) / (l**2 - e1))

        return no

    def calc_ne(l):
        l = l / 1000
        ne = a2 + (b2 / ((l**2) - c2)) + (d2*(l**2) / ((l**2) - e2))
        return ne

    # Calculate the refractive indices for the given wavelengths
    wavelengths = [lamb, 810]
    fig= plt.figure(figsize = (10, 5))
    l=np.linspace(wavelengths[1],wavelengths[1], 90)
    y=calc_no(l)
    plt.plot(y, label=f"no({wavelengths[1]})")

    l=np.linspace(wavelengths[0],wavelengths[0], 90)
    v=calc_no(l)
    plt.plot(v,label=f"no({wavelengths[0]})")

    l=np.linspace(wavelengths[0],wavelengths[0], 90)
    z=calc_ne(l)
    plt.plot(z,label=f"n_E({wavelengths[0]},90)")

    plt.xlabel('PM theta')
    plt.ylabel('Refractive index(n)')
    no=calc_no(wavelengths[0])
    ne=calc_ne(wavelengths[0])

    theta=np.linspace(0,3.14/2,90)
    s=((np.cos(theta))**2/no**2)+((np.sin(theta))**2/ne**2)
    nee=np.sqrt(1/s)
    plt.plot(nee,label=f"ne({wavelengths[0]},theta)")
    plt.grid(True, which='both', linestyle=':', color='red', linewidth=1.5, alpha=0.4)


    plt.legend()
    ne_values = [calc_ne(l) for l in wavelengths]
    no_values = [calc_no(l) for l in wavelengths]


    theta = (3) * np.pi / 180  # Convert 3 degrees to radians
    a = ((1 / ((no_values[1]**2) * (np.cos(theta)**2))) - (1 / no_values[0]**2)) / ((1 / ne_values[0]**2) - (1 / no_values[0]**2))
    delta_pm_ktp = np.arcsin(np.sqrt(a)) * 180 / np.pi

    # Print the results
    print(f"ne values KDP: {ne_values}")
    print(f"no values KDP: {no_values}")
    print(f"Phase matching angle for KDP: {delta_pm_ktp:.2f} degrees")
    
    return delta_pm_ktp , fig


# Author: umesh chandra joshi
# Created: 2023-02-11

# This code is licensed under the MIT License.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# """
# here are all the functions defined in the code:

#     normalize_state(state_vector): calculates the normalized state vector from the given state vector.

#     density_matrix(state_vector): calculates the density matrix from the given state vector.

#     eigenvalue(den): calculates the eigenvalues of a matrix.

#     state_rand(n): generates a random complex state vector.

#     density_rand(n): generates a random density matrix.

#     ppt(den, position): performs a bitwise operation on a square array called den

# """
# Title: normaliszed state vector from given Vector
# Description: Computes the normaliszed state vector vector

import numpy as np
from numpy import linalg as la

def normalize_state(state_vector):
    
    # Convert the input to a numpy array if it's not already one
    if not isinstance(state_vector, np.ndarray):
        state_vector = np.asarray(state_vector)
        
    if len(state_vector.shape) != 1:
        raise ValueError("Input must be a 1D numpy array.")
    
    # Calculate the norm of the state vector
    norm = np.linalg.norm(state_vector)
    
    # Normalize the state vector by dividing it by its norm
    normalized_state_vector = state_vector / norm
    
    return normalized_state_vector

# Author: umesh chandra joshi
# Created: 2023-02-11
# Title: Density Matrix Calculation from State Vector
# Description: Computes the density matrix of a given state vector



def density_matrix(state_vector):
    """
    Computes the density matrix of a given state vector.
    """
    # Convert the input into a numpy array if it is not already one
    state_vector = np.asarray(state_vector)
    
    # Check if the input is a numpy array
    if not isinstance(state_vector, np.ndarray):
        raise TypeError("Input must be a numpy array or a convertible object.")
    
    # Check if the input is a 1D numpy array
    if len(state_vector.shape) != 1:
        raise ValueError("Input must be a 1D numpy array.")
        
    state_vector = normalize_state(state_vector)
    
    # Calculate the outer product of the state vector and its conjugate
    return np.outer(state_vector, np.conj(state_vector))



def eigenvalue(den):
    eigen=la.eigvals(den)
#     print(np.round(eigen,6))
    return eigen

# Generating Random Density Matrix

def state_rand(n):
    """Generates a random complex state vector.
    
    Parameters:
        n (int): The length of the state vector.
    
    Returns:
        state_vector (numpy array): The complex state vector.
    """
    vec1 = np.random.rand(2**n)
    vec2 = np.random.rand(2**n)
    state_vector = vec1 + vec2 * 1j
    state_vecto = normalize_state(state_vector)
    
    return state_vecto
 

    return state_vector
  
def density_rand(n):
    """Generates a random density matrix.
    
    Parameters:
        n (int): The length of the state vector.
    
    Returns:
        density_random (numpy array): The random density matrix.
        trace (float): The trace of the density matrix.
    """
    state_vector = state_rand(n)
    density_random = density_matrix(state_vector)
    return density_random

# Description: The ppt function performs a bitwise operation on a square array called den.
# The function takes two inputs: den, a square 2D array and position, an integer.
# The function returns the result of the bitwise operation as a square 2D array.

import numpy as np
def ppt(den,n, position):
    if position >= n:
        return "Error: position must be less than n"

    nn = 2**n
    a = []# Initialize an empty list to store the result
    pp = n - position - 1 # Calculate the position to perform the bitwise operation
    list1 = [0] * nn # Initialize a list with nn zeros
    list2 = [0] * nn # Initialize a second list with nn zeros
    for i in range(0,nn):
        list1[i] = bin(i)[2:].zfill(n) # Convert each number in the list to binary and fill with zeros if necessary
        list2[i] = bin(i)[2:].zfill(n) # Convert each number in the list to binary and fill with zeros if necessary
    for i in range (0,nn):
        for j in range (0,nn):
            m = 0 # Initialize m
            n = 0 # Initialize n
            # If the bit at the specified position in list1 and list2 are different
            if list1[i][pp] != list2[j][pp]:
                c = int(list1[i][pp]) # Convert the bit from string to int
                if c == 0:
                    m = i + 2**position # Update m
                    n = j - 2**position # Update n
                    a.append(den[m][n]) # Append the value from den to the result list
                if c == 1:
                    m = i - 2**position # Update m
                    n = j + 2**position # Update n
                    a.append(den[m][n]) # Append the value from den to the result list
            else:

                a.append(den[i][j]) # If the bits are the same, append the value from den to the result list

                
    a = np.array(a).reshape(nn, nn) # Convert the result list to a square 2D numpy array

    
    return a # Return the result



# h=normalize_state([1,0])
# v=normalize_state([0,1])
# d=normalize_state([1,1])
# l=normalize_state([1,1j])

# mh=density_matrix(h)
# mv=density_matrix(v)
# md=density_matrix(d)
# ml=density_matrix(l)


# traceh=(mh@matrix).trace()
# L11=((traceh-N[0])**2)/(2*traceh)
# tracev=(mv@matrix).trace()
# L22=((tracev-N[1])**2)/(2*tracev)
# traced=(md@matrix).trace()
# L33=((traced-N[2])**2)/(2*traced)
# tracel=(ml@matrix).trace()
# L44=((tracel-N[3])**2)/(2*tracel)


def is_hermitian(matrix):
    """
    Check if the matrix is Hermitian.

    Parameters:
    matrix (numpy.ndarray): The matrix to be checked.

    Returns:
    bool: True if the matrix is Hermitian, False otherwise.
    """
    return np.allclose(matrix, matrix.conj().T)


def is_density_matrix(matrix):
    """
    Check if the matrix is a valid density matrix.

    A density matrix is a Hermitian positive semi-definite matrix with trace equal to 1.

    Parameters:
    matrix (numpy.ndarray): The matrix to be checked.

    Returns:
    bool: True if the matrix is a density matrix, False otherwise.
    """
    # Check if the matrix is Hermitian
    if not is_hermitian(matrix):
        print(0,"matrix is not Hermitian")
        return False

    # Check if the matrix is positive semi-definite
    eigenvalues= eigenvalue(matrix)
    
    if (eigenvalues < -0.000001).any():
        print(1,"matrix is not positive semi-definite")
        return False

    # Check if the trace of the matrix is equal to 1
    if np.trace(matrix) != 1+0j and  np.trace(matrix)<0.99999999999 and np.trace(matrix)>1.00000000002 :
        
        print(2,"trace of the matrix is notequal to 1")
        return False

    return True


import numpy as np
from scipy.optimize import minimize

# Define the Pauli matrices
sigmaeye = np.eye(2)
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
allsigma = [sigmaeye, sigmax, sigmay, sigmaz]

def qubit1_tomography(a, b, c, d):
    """
    This function calculates the density matrix from the given experimental data and performs maximum likelihood estimation optimization on the density matrix.
    The function returns the density matrix from the experimental data and the optimized density matrix.
    
    Parameters:
    a (float): the value for the first measurement
    b (float): the value for the second measurement
    c (float): the value for the third measurement
    d (float): the value for the fourth measurement
    
    Returns:
    tuple: A tuple containing two numpy arrays - the density matrix from the experimental data and the optimized density matrix.
    """
    # Normalize the experimental data
    N = np.array([a, b, c, d])
    N = N / (N[0] + N[1])

    # Calculate the values of s
    
    s = [1, 2 * N[2] - 1, 2 * N[3] - 1, N[0] - N[1]]

    # Calculate the density matrix from the experimental data
    den = np.zeros((2, 2))
    for i in range(0, 4):
        den = den + s[i] * allsigma[i]
    den = den / 2

    # Define the objective function for the optimization
    def objective_func(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        L1 = ((x[0]**2) + (x[2]**2) + (x[3]**2) - N[0])**2 / (2 * ((x[0]**2) + (x[2]**2) + (x[3]**2)))
        L2 = ((x[1]**2)-N[1])**2  / (2 * (x[1]**2))
        L3 = (((x[0]**2)+ (x[1]**2)+(x[2]**2)+(x[3]**2) +((2*x[1]*x[2]))) * 0.5 - N[2])**2/ (1 * (x[0]**2)+ (x[1]**2)+(x[2]**2)+(x[3]**2) +((2*x[1]*x[2])))
        L4 = (((x[0]**2)+(x[2]**2)+ (x[3]**2)+(x[1]**2) +(2*x[1]*x[3])) *0.5   - N[3])**2 / (((x[0]**2)+(x[2]**2)+(x[3]**2)+(x[1]**2) +(2*x[1]*x[3])))

        return np.real(L1 + L2 + L3 + L4)

    # Define the constraint for the optimization
    def cons2(x):
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 1

    x0 = np.array([1, 1, 2, 2])

    # Perform the optimization using the SLSQP method
    def min_obj_func(N, x0):
        cons = [{'type': 'eq', 'fun': cons2}]
        res = minimize(objective_func, x0, method='SLSQP', constraints=cons)

        t1, t2, t3, t4 = res.x

        T_d = np.array([[t1, 0], [t3 + (1j * t4), t2]])
        #print("Density Matrix After MLE Optimisation Algorithm")
        
        rho=np.round(np.conj(T_d).T @ T_d, 6)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(122, projection='3d')
        ax1= fig.add_subplot(121, projection='3d')
        x_data = np.array([0,1])
        y_data = np.array([0, 1])
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
        
        from qutip import Bloch
        b = Bloch()
        b.clear()
        from qutip import Bloch
        b = Bloch()
        v = [stokes_vector(den)[i] for i in range(1, 4)]
        b.add_vectors(v)
        v = [stokes_vector(rho)[i] for i in range(1, 4)]
        b.add_vectors(v)
        b.show()
        
        return den,rho
    return min_obj_func(N, x0)



def stokes_vector(density):
    """
    This function computes the Stokes vector given a density matrix.
    
    Parameters:
    density (np.ndarray): The density matrix for which the Stokes vector is to be computed
    
    Returns:
    np.ndarray: The Stokes vector
    
    """
    stokes = []
    
    # Define the Pauli matrices
    sigmaeye = np.eye(2)
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    allsigma = [sigmaeye, sigmax, sigmay, sigmaz]
    
    # Compute the elements of the Stokes vector
    stokes.append(1)
    for i in range(1,4):
        a = (allsigma[i] @ density).trace()
        stokes.append(a)
    
    return np.real(stokes)


def bloch_sphere(den):
    from qutip import Bloch
    b = Bloch()
    b.clear()
    v = [stokes_vector(den)[i] for i in range(1, 4)]
    b.add_vectors(v)
    b.show()
    return v

import math
from qutip import Bloch
import cmath

def grover(n,m):
    
    n = 2**n
    lam = m / n

    lam1 = math.sqrt(lam)
    lam2 = 1 - lam
    lam2 = math.sqrt(lam2)
    vec = np.array([lam2, lam1])

    den2 = np.outer(vec, vec)
    den = np.outer(vec, vec)
    print(vec)
    # Calculate the expectation values
    s0 = np.trace(den2)
    s1 = np.trace(np.array([[0, 1], [1, 0]]) @ den2)
    s2 = np.trace(np.array([[0, -1j], [1j, 0]]) @ den2)
    s3 = np.trace(np.array([[1, 0], [0, -1]]) @ den2)

    # Plot the Bloch sphere
    b = Bloch()
    b.clear()
    vec4 = [np.real(s1), np.real(s2), np.real(s3)]
    b.add_vectors(vec4)

    # Perform the rotation and plot the Bloch sphere again
    kopt = ((np.pi) / (4 * np.arcsin(lam1))) - 0.5
    kopt = np.round(np.ceil(kopt))
    kki = (4 * kopt) + 2
    x = (1 / lam1) * np.sin(np.pi / kki)
    theta = 2 * np.arcsin(x)


    z = np.pi* 1j
    result = cmath.exp(z)
    rot1 = np.array([[1, 0], [0, result]])

    result1 = cmath.exp(z)
    z = -1 * z
    result2 = cmath.exp(z)

    ii = np.array([[1, 0], [0, 1]])
    yes = (1 - result2) * den
    rot2 = result1*(ii - yes)

    for i in range(int(kopt)):
        vec =rot2 @ rot1 @ vec
      #  vec = rot2 @ vec

        den2 = np.outer(vec, vec)
        s0 = np.trace(den2)
        s1 = np.trace(np.array([[0, 1], [1, 0]]) @ den2)
        s2 = np.trace(np.array([[0, -1j], [1j, 0]]) @ den2)
        s3 = np.trace(np.array([[1, 0], [0, -1]]) @ den2)

        vec4 = [np.real(s1), np.real(s2), np.real(s3)]
        b.add_vectors(vec4)
    
    b.show()
    return vec

def concurrence(den):
    
    a = np.array([[0, -1j], [1j, 0]])

    st = np.kron(a, a)

    dent = den.conj()

    f = den @ st @ dent @ st

    ei = np.round(np.real(np.linalg.eigvals(f)),5)
    ei.sort()
    
    C_b=0
    C_b = np.sqrt(ei[3]) - np.sqrt(ei[0]) - np.sqrt(ei[1])  - np.sqrt(ei[2])  
    
    
    return np.real(max(0, C_b))




    """
    
    # This is a Python function called "partialtrace", which takes two parameters:
# - "den": a numpy array representing a density matrix of a quantum state. The function assumes that the state
is composed of qubits.
# - "b": a list of integers representing the indices of the qubits to be traced out from the density matrix.
# 
# The function calculates the partial trace of the density matrix with respect to the qubits in the "b" list.
The partial trace operation can be thought of as a way to trace out the degrees of freedom of a subsystem and obtain the reduced density matrix of the remaining subsystem.
#
# The function first calculates the number of qubits in the system, "n", and the number of qubits 
to be traced out, "nt". It then creates two lists:
# - "c": a list of the remaining qubits (i.e., those not in "b").
# - "lst": a list of all possible combinations of 0 and 1 of length "nt".
# Using these lists, the function calculates the indices of the traced out and remaining qubits,
and uses them to construct the reduced density matrix.
# 
# Finally, the function returns a numpy array representing the reduced density matrix.
The function also prints a message indicating which qubits were traced out.

    """

import numpy as np
import itertools

def partialtrace(den, b):
    
    # Get the number of qubits in the system
    n = int(np.log2(den.shape[0]))
    
    if len(b) > n:
        raise ValueError("The length of 'b' cannot be greater than the number of qubits in the system.")
    if any(q >= n for q in b):
        raise ValueError("The values in 'b' must be integers less than the number of qubits in the system.")

    # Create a list of the qubits to be traced out
    c = []
    for i in range(n):
        if i not in b:
            c.append(i)

    # Print a message indicating which qubits were traced out
    print("Partial trace is taken with respect to system", b)

    # Calculate the number of traced-out and remaining qubits
    nt = len(b)
    nl = n - nt

    # Calculate the dimensions of the reduced density matrix
    nnt = 2**nt
    nnl = 2**nl

    # Create a list of all possible combinations of 0 and 1 of length nt
    lst = list(itertools.product([0, 1], repeat=nt))

    # Calculate the indices of the traced-out qubits
    a = np.zeros((nnt,), dtype=int)
    for i in range(0, nnt):
        for j in range(0, nt):
            a[i] = a[i] + lst[i][nt-1-j]*(2**b[j])

    # Create a list of all possible combinations of 0 and 1 of length nl
    lst2 = list(itertools.product([0, 1], repeat=nl))

    # Calculate the indices of the remaining qubits
    d = np.zeros((nnl,), dtype=int)
    for i in range(0, nnl):
        for j in range(0, nl):
            d[i] = d[i] + lst2[i][nl-1-j]*(2**c[j])

    # Create an array to store the reduced density matrix
    aa = np.zeros((nnl, nnl), dtype=complex)

    # Calculate the reduced density matrix
    for i in range(0, nnt):
        for j in range(0, nnl):
            index1 = a[i] + d[j]
            for k in range(0, nnl):
                index2 = a[i] + d[k]
                aa[j][k] = aa[j][k] + den[index1][index2]

    return aa





import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Constants for calculate_n_z
# A_z = 2.12725
# B_z = 1.18431
# C_z = 5.14852e-2
# D_z = 0.6603
# E_z = 100.00507
# F_z = 9.68956e-3
A_z = 2.25411
B_z = 1.06543
C_z = 0.05486
D_z = 0.02140
# Constants for calculate_n_y
A_y = 2.19229
B_y = 0.83547
C_y = 0.04970
D_y = 0.01621

# Constants for calculate_n_x
A_x = 2.16747
B_x = 0.83733
C_x = 0.04611
D_x = 0.01713


# def calculate_n_z(wavelength):
#     wavelength = wavelength / 1000
#     n_z_squared = A_z + B_z / (1 - C_z / wavelength**2) + D_z / (
#         1 - E_z / wavelength**2) - F_z * wavelength**2
#     return n_z_squared**0.5

def calculate_n_z(wavelength):
    wavelength = wavelength / 1000
    n_z_squared = A_z + B_z / (1 - C_z / wavelength**2) + (D_z * wavelength**2)
    return n_z_squared**0.5


def calculate_n_y(wavelength):
    wavelength = wavelength / 1000
    n_y_squared = A_y + B_y / (1 - C_y / wavelength**2) + (D_y * wavelength**2)
    return n_y_squared**0.5


def calculate_n_x(wavelength):
    wavelength = wavelength / 1000
    n_x_squared = A_x + B_x / (1 - C_x / wavelength**2) + (D_x * wavelength**2)
    return n_x_squared**0.5


def n1_n2_z(wavelength):
    a = np.array([9.9587, 9.9228, -8.9603, 4.1010]) * 1e-6
    b = np.array([-1.1882, 10.459, -9.8136, 3.1481]) * 1e-8
    temp1 = 0
    temp2 = 0
    wavelength = wavelength / 1000
    for i in range(0, len(a)):
        temp1 = temp1 + a[i] / wavelength**i
        temp2 = temp2 + a[i] / wavelength**i
    return temp1, temp2

def deltanz(temp, wavelength):
    n1 = n1_n2_z(wavelength)[0]
    n2 = n1_n2_z(wavelength)[1]
    deln = n1 * (temp - 25) + n2 * (temp - 25)**2
    return deln


def n1_and_n2_y(wavelength):
    a = np.array([6.2897, 6.3061, -6.0629, 2.6486]) * 1e-6
    b = np.array([-0.14445, 2.2244 - 3.5770, 1.3470]) * 1e-8
    temp1 = 0
    temp2 = 0
    wavelength = wavelength / 1000
    for i in range(0, len(a)):
        temp1 = temp1 + a[i] / wavelength**i
        temp2 = temp2 + a[i] / wavelength**i
    return temp1, temp2

def deltany(temp, wavelength):
    n1 = n1_and_n2_y(wavelength)[0]
    n2 = n1_and_n2_y(wavelength)[1]
    deln = n1 * (temp - 25) + n2 * (temp - 25)**2
    return deln


def calculate_n_zt(wavelength, temperature):
    """
    Calculate the refractive index n_z with a temperature-dependent correction.

    Args:
        wavelength (float): The wavelength of light in nanometers.
        temperature (float, optional): The temperature in degrees Celsius (default is 25°C).

    Returns:
        float or None: The refractive index n_z if it's a real number; otherwise, None.
    """
    n_z = calculate_n_z(wavelength) + deltanz(temperature, wavelength)
    return n_z


def calculate_n_yt(wavelength, temperature):
    n_y = calculate_n_y(wavelength) + deltany(temperature, wavelength)
    return n_y


def calculate_n_xt(wavelength, temperature):
    n_x = calculate_n_x(wavelength) + deltany(temperature, wavelength)
    return n_x

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def phase_matching_temp(a):
    
    tvalues = np.linspace(20, 80, 10)

    wavelength1 = a
    wavelength2 = wavelength1 * 2
    mode = 1
    periods = [3.415, 3.425, 3.435, 5]
    fig = make_subplots(rows=2,
                        cols=2,
                        subplot_titles=('Period: 1', 'Period: 2', 'Period: 3',
                                        'Period: 4'))

    for i, period in enumerate(periods, start=1):
        nzvaluetemp = []
        nyvaluetemp = []

        for t in tvalues:
            n1 = calculate_n_zt(wavelength1, t) / (wavelength1 / 1000)
            n2 = (calculate_n_zt(
                wavelength2, t)) / (2 * wavelength1 / 1000) + (calculate_n_zt(
                    wavelength2, t)) / (2 * wavelength1 / 1000) + (mode) / (period)

            nzvaluetemp.append(n1)
            nyvaluetemp.append(n2)

        trace_n1 = go.Scatter(x=tvalues,
                              y=nzvaluetemp,
                              mode='lines+markers',
                              name=f'kpump (Period: {period}mu m)',
                              line=dict(color='red'))
        trace_n2 = go.Scatter(x=tvalues,
                              y=nyvaluetemp,
                              mode='lines+markers',
                              name=f'k(s,i,period) (Period: {period}mu m)',
                              line=dict(color='blue'))

        row = 1 if i <= 2 else 2
        col = i if i <= 2 else i - 2
        fig.add_trace(trace_n1, row=row, col=col)
        fig.add_trace(trace_n2, row=row, col=col)

    fig.update_layout(
        title=
        'TYPE 0 QPM (Collinear) poling period vs Temperature (lambda_pump = 405nm, lambda(s,i)=810) ',
        xaxis=dict(title='Temperature'),
        yaxis=dict(title='delta_k (Momentum)'),
        template="plotly_dark",
        width=1000,  # Set the width
        height=500  # Set the height
    )
#     fig.show()
    return fig
import numpy as np
import plotly.graph_objs as go
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt  # Add this import

def degeneracy_graph_ppktp(a,b,period):
    fixed_period = period
    temperature_range = np.linspace(a, b, 100)
    wavelength1 = 405
    mode = 1

    wavelength2_values = []
    wavelength3_values = []
    deltak = []


    def calculate_wavelength2(temperature):
        def objective_function(wavelength2):
            wavelength3 = (1 / wavelength1 - 1 / wavelength2) ** -1
            n1 = calculate_n_zt(wavelength1, temperature) / (wavelength1 / 1000)
            n2 = (calculate_n_zt(wavelength2, temperature)) / (2 * wavelength1 / 1000) + (
                calculate_n_zt(wavelength3, temperature)) / (2 * wavelength1 / 1000) + (
                    mode) / (fixed_period)
            return np.abs(n1 - n2)

        result = minimize_scalar(objective_function, bounds=(100, 4850), method='bounded')
        deltak.append(objective_function(result.x))
        return result.x

    def calculate_wavelength3(temperature, wavelength2):
        wavelength3 = (1 / wavelength1 - 1 / wavelength2) ** -1
        return wavelength3

    # Calculate Wavelength2 and Wavelength3 for each temperature in the range
    for temperature in temperature_range:
        wavelength2_equal = calculate_wavelength2(temperature)
        wavelength2_values.append(wavelength2_equal)
        wavelength3_equal = calculate_wavelength3(temperature, wavelength2_equal)
        wavelength3_values.append(wavelength3_equal)

    trace_wavelength2 = go.Scatter(
        x=temperature_range,
        y=wavelength2_values,
        mode='markers',
        name=f'Wavelength(signal)',
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=6)
    )
    trace_wavelength3 = go.Scatter(
        x=temperature_range,
        y=wavelength3_values,
        mode='markers',
        name=f'Wavelength(idler)',
        line=dict(color='red'),
        marker=dict(symbol='cross', size=4
        )
    )

    layout = go.Layout(
        title=f'Wavelength(signal) and Wavelength(idler) vs. Temperature (Fixed Period {fixed_period})',
        xaxis=dict(title='Temperature (°C)'),
        yaxis=dict(title='Wavelength (nm)'),
        width=1000,
        height=500
    )

    fig = go.Figure(data=[trace_wavelength2, trace_wavelength3], layout=layout)

#     fig.show()
    return fig


# degeneracy_graph_ppktp(25,60,3.425)
def concurrence(den):
    
    a = np.array([[0, -1j], [1j, 0]])

    st = np.kron(a, a)

    dent = den.conj()

    f = den @ st @ dent @ st

    ei = np.round(np.real(np.linalg.eigvals(f)),8)
    ei.sort()
    
    C_b=0
    C_b = np.sqrt(ei[3]) - np.sqrt(ei[0]) - np.sqrt(ei[1])  - np.sqrt(ei[2])  
    
    
    return np.real(max(0, C_b))




def amplitude(a1):

    b=math.sqrt(1-a1)
    k1=np.array([[1, 0], [0,b]])
    e1=np.kron(k1,sigmaeye)
    sconj1=np.conj(e1).T
    a2=a1
    a2=math.sqrt(a2)
    e2=np.kron(np.array([[0, a2], [0,0]]),sigmaeye)
    sconj2=np.conj(e2).T

    final=(e1 @ density_matrix([1,0,0,1]) @ sconj1)    + (e2 @ density_matrix([1,0,0,1]) @ sconj2)
    
    c=concurrence(final)
    
    ca=[]

    for i in np.arange(0,1,0.01):
        ca.append(amplitude(i))


    plt.scatter(np.arange(0, 1, 0.01), ca,s=5)
    plt.xlabel('lambda')
    plt.ylabel("concurrence(c)")
    plt.title(" Concurrence at different values of lambda in (Amplitude Damping channel)")
    plt.show()
    
    return c


from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def qubit2_tomography(N):
#     N=np.array([np.random.randint(100,1000) for x in range (16)])
#     N = np.array([3708, 77, 1791, 2048, 51, 3642, 2096, 1926, 1766, 1914, 1713, 3729, 2017,1709,3686,2404])#exact
#     print(N[0] + N[1] + N[4] + N[5])
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



   
    print(is_density_matrix(rho))
    print( " Concurrence ",concurrence(rho))

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
    # plt.show()
    
    
    return np.round(rho,3) , fig

def nxz(theta, wavelength, temperature):

    nxzn = calculate_n_xt(wavelength, temperature) * calculate_n_zt(
        wavelength, temperature)
    nxzd = np.sqrt((calculate_n_xt(wavelength, temperature)**2 *
                    (np.cos(theta)**2)) +
                   (calculate_n_zt(wavelength, temperature))**2 *
                   (np.sin(theta)**2))
    return nxzn / nxzd


def main():
    sidebar_style = """
        background-color: #3498db; /* Use your desired color code */
        padding: 10px;
    """
    st.markdown(
        f"""
        <style>
            .sidebar {{
                {sidebar_style}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page Navigation
    page_icons = {
        "Introduction": "🏠",
        "Quantum State Tomography": "",
        "SPDC Sources": "",
        "Phase Matching Angle BBO,KDP": "",
        "Compensation Crystal Length": "",
        "PPKTP QPM Type(0 and 2)": "",
        "Authors": "",
    }

    page = st.sidebar.radio(
        "Select a Page",
        list(page_icons.keys()),
        format_func=lambda page_name: f"{page_icons[page_name]} {page_name}",
    )

    if page == "Introduction":
        introduction_page()
    elif page == "Quantum State Tomography":
        software_list_page()
    elif page == "SPDC Sources":
        projects()
    elif page == "Phase Matching Angle BBO,KDP":
        phase_matching_page()
    elif page == "Compensation Crystal Length":
        phase_matching_page()
    elif page == "PPKTP QPM Type(0 and 2)":
        phase_matching_ppktp()
    elif page == "Authors":
        phase_matching_type12()
if __name__ == "__main__":
    main()
