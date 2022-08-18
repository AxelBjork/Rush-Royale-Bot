import socket
# import thread module
from _thread import *
import threading
import time
import os
from subprocess import check_output, Popen, DEVNULL


# Connects to a target IP and port, if port is open try to connect adb
def connect_port(ip, port, batch, open_ports):
    for tar_port in range(port, port + batch):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, tar_port))
        if result == 0:
            open_ports[tar_port] = 'open'
            # Make it Popen and kill shell after couple seconds
            p = Popen(f'.scrcpy\\adb connect {ip}:{tar_port}', shell=True)
            time.sleep(3)  # Give real client 3 seconds to connect
            p.terminate()
    return result == 0


# Attemtps to connect to ip over every port in range
# Returns device if found
def scan_ports(target_ip, port_start, port_end, batch=3):
    threads = []
    open_ports = {}
    port_range = range(port_start, port_end, batch)
    socket.setdefaulttimeout(0.01)
    print(f"Scanning {target_ip} Ports {port_start} - {port_end}")
    # Create one thread per port
    for port in port_range:
        thread = threading.Thread(target=connect_port, args=(target_ip, port, batch, open_ports))
        threads.append(thread)
    # Attempt to connect to every port
    for thread in threads:
        thread.start()
    # Join threads
    print(f'Started {len(port_range)} threads')
    for thread in threads:
        thread.join()
    # Get open ports
    port_list = list(open_ports.keys())
    print(f"Ports Open: {port_list}")
    deivce = get_adb_device()
    return deivce


# Check if adb device is already connected
def get_adb_device():
    devList = check_output('.scrcpy\\adb devices', shell=True)
    devListArr = str(devList).split('\\n')
    # Check for online status
    deivce = None
    for client in devListArr[1:]:
        client_ip = client.split('\\t')[0]
        if 'device' in client:
            deivce = client_ip
            print("Found ADB device! {}".format(deivce))
        else:
            Popen(f'.scrcpy\\adb disconnect {client_ip}', shell=True, stderr=DEVNULL)
    return deivce


def get_device():
    p = Popen([".scrcpy\\adb", 'kill-server'])
    p.wait()
    p = Popen('.scrcpy\\adb devices', shell=True, stdout=DEVNULL)
    p.wait()
    # Check if adb got connected
    device = get_adb_device()
    if not device:
        # Find valid ADB device by scanning ports
        device = scan_ports('127.0.0.1', 48000, 65000)
    if device:
        return device