# The DE1-SoC oscillator is 50 MHz, but use a 100 MHz implementation target
# so synthesis and fitting continue optimizing instead of stopping as soon as
# the board clock requirement is met.  This is a conservative timing target;
# using it above 50 MHz on hardware still requires a PLL-generated core clock.
create_clock -name clk -period 10.000 [get_ports {clk}]

# Account for device and clock-network uncertainty in setup/hold analysis.
derive_clock_uncertainty
