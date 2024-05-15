# main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from meningitis.simulation import vac_prob, make_sim, vac_prob_age, treat_prob

def main():
    st.title("Meningitis Simulation")
    
    st.sidebar.title("Simulation Parameters")
    n_agents = st.sidebar.number_input("Number of Agents", value=2000)
    n_timesteps = st.sidebar.number_input("Number of Timesteps", value=100)
    prob = st.sidebar.slider("Vaccination Probability", 0.0, 1.0, 0.5)
    imm_boost = st.sidebar.number_input("Immunity Boost", value=2.0)
    use_vaccine = st.sidebar.checkbox("Use Vaccine", value=True)
    
    if st.sidebar.button("Run Simulation"):
        sim = make_sim(seed=1, n_timesteps=n_timesteps, prob=prob, imm_boost=imm_boost, use_vaccine=use_vaccine)
        sim.run()
        
        st.write("Simulation Completed")
        fig = sim.diseases.meningitis.plot()
        st.pyplot(fig)

    st.sidebar.title("Different Vaccination Proportions")
    if st.sidebar.button("Run Different Vaccination Proportions"):
        vac_prob()

    st.sidebar.title("Different Vaccination Proportions by Age")
    if st.sidebar.button("Run Different Vaccination Proportions by Age"):
        vac_prob_age()

    st.sidebar.title("Treatment Simulation")
    if st.sidebar.button("Run Treatment Simulation"):
        treat_prob()

if __name__ == "__main__":
    main()
