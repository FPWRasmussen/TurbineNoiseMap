import numpy as np
import threading
import time

n_threads = 3


array = np.zeros([5, n_threads])
# Define a function for the thread
def update_array(array, n):
    array[:,n] = n+1 

# Create two threads as follows

for n in np.arange(0, n_threads):
    threading.Thread(target=update_array,  args= (array, n)).start()
    print(array)

