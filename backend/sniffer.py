import socket
import threading
from scapy.all import sniff, IP, TCP

HOST = "127.0.0.1"
PORT = 9999

# Function to send packets
def generate_traffic():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    for i in range(5):  # Send 5 messages
        message = f"Generated Packet {i}"
        client_socket.sendall(message.encode())
        print(f"Sent: {message}")

    client_socket.close()

# Function to capture packets
def capture_packets(packet):
    if packet.haslayer(IP):
        print(f"Captured Packet: {packet[IP].src} -> {packet[IP].dst}, Size: {len(packet)} bytes")

# Run both in parallel
threading.Thread(target=generate_traffic).start()
sniff(filter="ip", prn=capture_packets, count=5)
