#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import streamlit as st
import numpy as np
def B92(a,x,y,y1,y2,z1,z2): 
    
    # Randomly choose 50% of 0s in array_0s to replace with 2

    # Create arrays with 2s at positions of 0s and 1s in the original array
    array_0s = np.where(a == 1, 2, 0)
    array_1s = np.where(a == 0, 2, 1)

    original0=array_0s
    original1=array_1s
    print(original0)
    print(original1)

    # Identify the positions of 0s in array_0s
    zero_positions_0s = np.where(array_0s == 0)[0]
    # Identify the positions of 1s in array_1s
    zero_positions_1s = np.where(array_1s == 1)[0]
    replace_indices_0s = np.random.choice(zero_positions_0s, size=int(x* len(zero_positions_0s)), replace=False)
    array_0s[replace_indices_0s] = 2

    # Randomly choose 50% of 1s in array_1s to replace with 2
    replace_indices_1s = np.random.choice(zero_positions_1s, size=int(x * len(zero_positions_1s)), replace=False)
    array_1s[replace_indices_1s] = 2

    print("BS1 LOSS ",x*100,"%")
    print(array_0s)
    print("BS1 LOSS ",x*100,"%")
    print(array_1s)


    # Identify the positions of 0s in array_0s
    zero_positions_0s = np.where(array_0s == 0)[0]
    # Identify the positions of 1s in array_1s
    zero_positions_1s = np.where(array_1s == 1)[0]

    replace_indices_0s = np.random.choice(zero_positions_0s, size=int(y * len(zero_positions_0s)), replace=False)
    array_0s[replace_indices_0s] = 1


    replace_indices_1s = np.random.choice(zero_positions_1s, size=int(y * len(zero_positions_1s)), replace=False)
    array_1s[replace_indices_1s] = 0

    print("Polarization Change in channel",y*100,"%")
    print(array_0s)
    print("Polarization Change in channel",y*100,"%")
    print(array_1s)
    
    # Identify the positions of 0s in array_0s
    zero_positions_0s = np.where(array_0s == 0)[0]
    # Identify the positions of 1s in array_1s
    zero_positions_1s = np.where(array_1s == 1)[0]
    replace_indices_0s = np.random.choice(zero_positions_0s, size=int(x* len(zero_positions_0s)), replace=False)
    array_0s[replace_indices_0s] = 2

    # Randomly choose 50% of 1s in array_1s to replace with 2
    replace_indices_1s = np.random.choice(zero_positions_1s, size=int(x * len(zero_positions_1s)), replace=False)
    array_1s[replace_indices_1s] = 2

    print("BS2 LOSS ",x*100,"%")
    print(array_0s)
    print("BS2 LOSS ",x*100,"%")
    print(array_1s)
    
    # Identify the positions of 0s in array_0s
    zero_positions_0s = np.where(array_0s == 0)[0]
    # Identify the positions of 1s in array_1s
    zero_positions_1s = np.where(array_1s == 1)[0]

    replace_indices_0s = np.random.choice(zero_positions_0s, size=int(y1 * len(zero_positions_0s)), replace=False)
    array_0s[replace_indices_0s] = 1

    # Randomly choose 50% of 1s in array_1s to replace with 2
    replace_indices_1s = np.random.choice(zero_positions_1s, size=int(y2 * len(zero_positions_1s)), replace=False)
    array_1s[replace_indices_1s] = 0

    print("Polarization Change in route1",y1*100,"%")
    print(array_0s)
    print("Polarization Change in route 2",y2*100,"%")
    print(array_1s)
    
    # Identify the positions of 0s in array_0s
    zero_positions_0s = np.where(array_0s == 0)[0]
    # Identify the positions of 1s in array_1s
    zero_positions_1s = np.where(array_1s == 1)[0]

    # Randomly choose 50% of 0s in array_0s to replace with 2
    replace_indices_0s = np.random.choice(zero_positions_0s, size=int(z1 * len(zero_positions_0s)), replace=False)
    array_0s[replace_indices_0s] = 2

    # Randomly choose 50% of 1s in array_1s to replace with 2
    replace_indices_1s = np.random.choice(zero_positions_1s, size=int(z2 * len(zero_positions_1s)), replace=False)
    array_1s[replace_indices_1s] = 2

    print("Detector loss1",z1*100,"%")
    print(array_0s)
    print("Detector loss2",z2*100,"%")
    print(array_1s)
    
    # Create a new array based on the specified conditions
    result_array = np.where((array_0s == 0) | (array_1s == 0), 0, np.where((array_0s == 1) | (array_1s == 1), 1, 2))
    print("Result Array:")
    print(result_array)
    
    return array_0s,array_1s,result_array

def calculate_error_rate(original_array, reconstructed_array):
    return np.sum(original_array != reconstructed_array) / len(original_array)

import numpy as np

def text_to_binary_array(text):

    binary_string = ''.join(format(ord(char), '08b') for char in text)
    binary_array = np.array([int(bit) for bit in binary_string])
    return binary_array

def binary_array_to_text(binary_array):

    binary_string = ''.join(str(bit) for bit in binary_array)
    text = ''.join(chr(int(binary_string[i:i+8], 2)) for i in range(0, len(binary_string), 8))
    return text

# Compare result_array and original array a
def error_rate_raw(a,result_array):
    mismatched_positions = np.where(result_array != a)[0]
    # Calculate error rate
    error_rate = len(mismatched_positions) / len(a)
    print("\nError Rate:")
    print(error_rate)
    return error_rate

def error_rate_no_BS(a,result_array):
    exclude_positions = np.where((result_array == 2) | (a == 2))[0]
    mismatched_positions = np.where((result_array != a) & (~np.isin(np.arange(len(a)), exclude_positions)))[0]
    error_rate = len(mismatched_positions) / (len(a) - len(exclude_positions))
    print("\nError Rate (Excluding Loss of BS):")
    print(error_rate)
    return error_rate

def error_rate_post(b,result_array,repeat):
    ab = result_array.reshape(-1, repeat)
    post = np.array([
        0 if np.sum(group == 0) >= np.sum(group == 1) else 1
        for group in ab
    ])
    ac = post.reshape(-1, repeat)
    post2 = np.array([
        0 if np.sum(group == 0) >= np.sum(group == 1) else 1
        for group in ac
    ])
    poste=post2
    error_rate = calculate_error_rate(poste, b)
    print("\nError Rate post process:")
    print(error_rate)
    print("Retrieved array",poste)
    print("output",binary_array_to_text(poste))
    return error_rate , poste , binary_array_to_text(poste)

def message(input_text,repeat):
    binary_representation = text_to_binary_array(input_text)
    b=binary_representation
    print(b)
    aa=[]
    for i in range(0,len(b)):
        if b[i]==0:
            for i in range(0,repeat):
                aa.append(0)
        else:
            for i in range(0,repeat):
                aa.append(1)

    aaa=[]
    for i in range(0,len(aa)):
        if aa[i]==0:
            for i in range(0,repeat):
                aaa.append(0)
        else:
            for i in range(0,repeat):
                aaa.append(1)
                
    return np.array(aaa),np.array(b)


def main():
    st.title("Quantum Key Distribution Simulation B92")

    # Text input for the original text
    input_text = st.text_input("Enter the message to send:", "Quantum Key Distribution Simulation B92 001010101001010")
    # Create two columns for sliders
    col1, col2 = st.columns(2)

    # Sliders for various parameters
    with col1:
        repeat = st.slider("Select the repetition factor:", min_value=1, max_value=30, value=15)
        BS1_loss = st.slider("BS1 Loss:", 0.0, 1.0, 0.3, step=0.01)
        BS2_loss = st.slider("BS2 Loss:", 0.0, 1.0, 0.3, step=0.01)
        channel_dph = st.slider("Channel phase damping coeff.:", 0.0, 1.0, 0.1, step=0.01)

    with col2:
        analys1_ext = st.slider("Analyzer1 Extinction:", 0.0, 1.0, 0.02, step=0.01)
        analys2_ext = st.slider("Analyzer2 Extinction:", 0.0, 1.0, 0.02, step=0.01)
        d1_eff = st.slider("Detector1 Efficiency:", 0.0, 1.0, 0.2, step=0.01)
        d2_eff = st.slider("Detector2 Efficiency:", 0.0, 1.0, 0.2, step=0.01)

    # Button to trigger the simulation
    if st.button("Run Simulation"):
        a,b = message(input_text, repeat)
        array_0s, array_1s, result_array = B92(a, BS1_loss, channel_dph, analys1_ext, analys2_ext, d1_eff, d2_eff)

        # Display results
        st.text("Original Text: {}".format(input_text))
        #st.text("Encoded Array with 0s: {}".format(array_0s))
        #st.text("Encoded Array with 1s: {}".format(array_1s))
        st.text("Encoded bits: {}".format(a))
        st.text("Received bits: {}".format(result_array))

        # Create two columns for results
        col3, col4 = st.columns(2)

        with col3:
            error_rate_raw_value = error_rate_raw(a, result_array)
            st.text("Error Rate (Raw): {}".format(error_rate_raw_value))

        with col4:
            error_rate_no_BS_value = error_rate_no_BS(a, result_array)
            st.text("Error Rate (No BS): {}".format(error_rate_no_BS_value))

        # Display Error Rate (Post Process) in a single column
        error_rate_post_value, post_result_array, output = error_rate_post(b, result_array, repeat)
        st.text("Error Rate (Post Process): {}".format(error_rate_post_value))
        st.text("Recieved Message: {}".format(output))

if __name__ == "__main__":
    main()

