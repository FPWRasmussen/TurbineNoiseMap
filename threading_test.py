import numpy as np
import threading
import time

threads = 10
start_time = time.perf_counter()
array = np.zeros(9999999)
# Define a function for the thread

def update_array(array):
    for i in range(len(array)):
        array[i] = array[i-1]+np.sqrt(i)



threads_list = []
for i in range(threads):
    t = threading.Thread(target=update_array,  args= (array,))
    t.start()
    threads_list.append(t)

for i in range(threads):
    t.join()



end_time = time.perf_counter()

total_time = end_time - start_time
print(total_time)