from model import Meningitis
import starsim as ss

def init_sim_data():
    """This function recieves parameters, performs initial simulation and returns the simulation data in json format"""
    meningitis=Meningitis()
    start=int(input("Enter start year: "))
    end=int(input("Enter end year: "))
    init_population=int(input("Enter initial population: "))
    pars = {
    'networks': {'type': 'random'},
    'start': start,
    'end': end,
    'dt': 1,
    'verbose': 0,
    'n_agents': init_population
}
    sim = ss.Sim(pars, diseases=meningitis)
    sim.run()
    initial_data = sim.export_df().to_json()
    return initial_data
    
data=init_sim_data() 
# print(data) 