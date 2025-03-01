import socket
import time

HOST = "127.0.0.1"  # Localhost
PORT = 9999         # Must match the server port

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

for i in range(10):  # Send 10 messages
    message = f"Packet {i}"
    client_socket.sendall(message.encode())
    print(f"Sent: {message}")
    time.sleep(1)

client_socket.close()
