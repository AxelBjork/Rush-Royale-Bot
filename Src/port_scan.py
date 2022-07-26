import socket
# import thread module
from _thread import *
import threading
import os
from subprocess import check_output
import configparser


# Connects to a target IP and port, if port is open try to connect adb
def connect_port (ip,port,open_ports):
    config = configparser.ConfigParser()
    config.read('config.ini')
    scrcpy_path=config['bot']['scrcpy_path']
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex((ip,port))
    if result == 0:
        open_ports[port]='open'
        os.system(f'{os.path.join(scrcpy_path,"adb")} connect 127.0.0.1:{port}')
    return result == 0

# Attemtps to connect to ip over every port in range
# Returns device if found
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
    # Also try default 5555 port
    threads.append(threading.Thread(target=connect_port, args=(target_ip, 5555, open_ports)))
    # Attempt to connect to every port
    for i in range(len(port_range)):
        threads[i].start()
    # Join threads
    for i in range(len(port_range)):
        threads[i].join()
    # Get open ports
    port_list = list(open_ports.keys())
    print(f"Ports Open: {port_list}")
    deivce = get_adb_device()
    return deivce

# Check if adb device is already connected 
def get_adb_device():
    config = configparser.ConfigParser()
    config.read('config.ini')
    scrcpy_path=config['bot']['scrcpy_path']
    devList = check_output(f'{os.path.join(scrcpy_path,"adb")} devices')
    devListArr = str(devList).split('\\n')
    # Check for online status
    deivce = None
    for client in devListArr[1:]:
        client_ip = client.split('\\t')[0]
        if 'device' in client:
            deivce = client_ip
            print("Found ADB device! {}".format(deivce))
        else:
            os.system(f'{os.path.join(scrcpy_path,"adb")} disconnect {client_ip}')
    return deivce


def get_device():
    # Check if adb already connected
    device = get_adb_device()
    if not device:
        # Find valid ADB device by scanning ports
        device=scan_ports('127.0.0.1',48000,65000)
    if device:
        return device