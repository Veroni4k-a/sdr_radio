import adi
sdr =adi.Pluto("ip:192.168.2.1")

print(f"RX Freq: {sdr.rx_lo / 1e6} MHz")

print(f"TX Freq: {sdr.tx_lo /1e6} MHz")

print(f"Sample Rate: {sdr.sample_rate /1e6} Mps")

print(f"Bandwidth: {sdr.rx_rf_bandwidth /1e6} MHz")
"""
RX Freq: 5864.999998 MHz
TX Freq: 2450.0 MHz
Sample Rate: 10.0 Mps
Bandwidth: 4.8 MHz

"""
