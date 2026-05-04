import pickle
from saap import Trajectory, Generator, ConstPowerLoad
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    # Open daopf_times.pkl

    daopf_times = {}
    with open("daopf_times.pkl", "rb") as f:
        daopf_times = pickle.load(f)

    admm_times = {}

    with open("admm_times.pkl", "rb") as f:
        admm_times = pickle.load(f)
    
    da_opf_num_buses = [key for key in daopf_times.keys()]
    admm_num_buses = [key for key in admm_times.keys()]

    daopf_list = [daopf_times[key] for key in daopf_times.keys()]
    admm_sequential_list = [admm_times[key]['BusBehaviours (sequential)']['time'] for key in admm_times.keys()]
    admm_parallel_list = [admm_times[key]['BusBehavioursParallel (threaded)']['time'] for key in admm_times.keys()]
    

    print(daopf_list)
    print(admm_sequential_list)
    print(admm_parallel_list)
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    axs.plot(admm_num_buses[:-1], daopf_list[0:-3], marker='o', label='Fully Centralized')
    axs.plot(admm_num_buses[:-1], admm_sequential_list[:-1], marker='v', label='ADMM (sequential)')
    axs.plot(admm_num_buses[:-1], admm_parallel_list[:-1], marker='s', label='ADMM (parallel)')
    axs.set_xlabel('Number of Buses')
    axs.set_ylabel('Run Time (s)')
    axs.legend()
    axs.grid(True)
    plt.show()

