import numpy as np
from shadow_map_calculator import shadow_map_calculator
import multiprocessing as mp
import os


def shadow_map_multiprocessing(processes, times, map_data, turbines):
    """
    INPUTS :
        process : number of processes
        times : full list of times
        map_data : map_data method
        turbines : turbines method
        
    
    """

    process_list = []
    for process in range(processes):
        time = times[process::processes]
        p = mp.Process(target=shadow_map_calculator,  args= (time, map_data, turbines, process))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    shadow_map = np.zeros(map_data.shape)
    
    for p in range(processes):
        file_name = f"temp/shadow_map_temp{p}.txt"
        try:
            temp_array = np.loadtxt(file_name)
            shadow_map += temp_array
            os.remove(file_name)
        except:
            print("File not found")
        
    # total_hours = (times[-1]-times[0]).total_seconds()/(60*60)
    # shadow_map = shadow_map/total_hours
    
    shadow_map /= 60
    
    np.savetxt("temp/shadow_map_temp.txt", shadow_map)

        
    return shadow_map

