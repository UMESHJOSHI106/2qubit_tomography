#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
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

st.title('test')

