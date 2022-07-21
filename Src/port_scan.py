import socket
# import thread module
from _thread import *
import threading
import os
from subprocess import check_output

# Connects to a target IP and port
def connect_port (ip,port,open_ports):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex((ip,port))
    if result == 0:
        open_ports[port]='open'
    return result == 0

# Attemtps to connect to ip over every port in range
# Returns list of open ports
def scan_ports(target_ip,port_start,port_end):
    threads = []
    open_ports = {}
    port_range = range(port_start, port_end)
    socket.setdefaulttimeout(0.01)
    print(f"Scanning {target_ip} Ports {port_start} - {port_end}")
    # Create one thread per port
    for port in port_range:
        thread = threading.Thread(target=connect_port, args=(target_ip, port, open_ports))
        threads.append(thread)
    # Attempt to connect to every port
    for i in range(len(port_range)):
        threads[i].start()
    # Join threads
    for i in range(len(port_range)):
        threads[i].join()
    # Get open ports
    port_list = list(open_ports.keys())
    print(f"Ports Open: {port_list}")
    return port_list

# Attemtps to connect ADB over every port provided
def scan_ADB(port_list):
    # Try to connect every open port in range
    for device in port_list:
        os.system(f'adb connect 127.0.0.1:{device}')
    # Check all connected adb devices
    devList = check_output('adb devices')
    devListArr = str(devList).split('\\n')
    # Check for online status
    for client in devListArr[1:]:
        if 'device' in client:
            # split IP from status and store device
            deivce = client.split('\\t')[0]
            print("Found ADB device! {}".format(deivce))
            return deivce
    return None

def get_device():
    # Find valid ADB device
    port_list=scan_ports('127.0.0.1',48000,65000)
    device = scan_ADB(port_list)
    return device