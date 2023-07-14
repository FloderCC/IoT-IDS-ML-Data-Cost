import os
import threading
import time

# resource usage measurement section
thread_running = cpu_thread = start_time = avg_cpu_percentage = None

#measure interval
measure_interval = 1e-05

# Function to continuously measure CPU percentage

def measure_cpu():
    global avg_cpu_percentage, measure_interval

    # Multi-platforms. Bad for small pieces of codes
    # cpu_percentages = []
    # while thread_running:
    #     cpu_percentage = psutil.cpu_percent(interval=0.00001, percpu=False)
    #     cpu_percentages.append(cpu_percentage)
    # avg_cpu_percentage = sum(cpu_percentages) / len(cpu_percentages)

    # linux. Testing
    cpu_count = os.cpu_count()
    cpu_load_sum = 0
    cpu_load_quantity = 0
    while thread_running:
        cpu_load_sum += [x / cpu_count for x in os.getloadavg()][-1]
        cpu_load_quantity += 1
        time.sleep(measure_interval)
    avg_cpu_percentage = cpu_load_sum / max(cpu_load_quantity, 1) * 100


def monitor_tic(measure_int=1e-05):
    global thread_running, cpu_thread, start_time, measure_interval

    measure_interval = measure_int
    thread_running = True
    cpu_thread = threading.Thread(target=measure_cpu)
    cpu_thread.start()
    start_time = time.perf_counter()


def monitor_toc():
    global thread_running, cpu_thread, start_time

    # Stop measuring time
    end_time = time.perf_counter()

    # Stop the CPU measurement thread
    thread_running = False
    cpu_thread.join()

    # Calculate the elapsed time
    return avg_cpu_percentage, end_time - start_time
