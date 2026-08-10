# Primary DE1-SoC FPGA fabric clock: 50 MHz (20 ns period).
create_clock -name clk -period 20.000 [get_ports {clk}]

# Account for device and clock-network uncertainty in setup/hold analysis.
derive_clock_uncertainty
