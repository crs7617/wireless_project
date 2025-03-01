from scapy.all import sniff, IP, TCP

def capture_packets(packet):
    if packet.haslayer(IP):
        print(f"Packet: {packet[IP].src} -> {packet[IP].dst}, Size: {len(packet)} bytes")

# Sniff packets from localhost
sniff(filter="ip", prn=capture_packets, count=10)
