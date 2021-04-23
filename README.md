# What is it?

**DMSS** (**D**ummy **M**odbus **S**lave **S**erver) is a simple Python tool that can be used co create 
MODBUS RTU server on host machine by using USB Serial Port to RS485 adapter.

# Features
*   possibilty to simulate real **MODBUS RTU** device
*   simple configuration using single **JSON** file
*   possibilty to specify serial port connection parameters from command line
*   possibilty to specify multiple values for single register/coil/input that will change while the program is running which can simulate real life device behaviour. Values in registers will change with specified refresh frequency. (default f = 1Hz)
*   possibilty to specify values for register in different formats (decimal, floating point, binary, hex)

# Usage
To run program please type following command:
```text
app/dmss.py -p /dev/tty* -b BAUDRATE -d DATABITS -n PARITY -s STOP_BITS -c configs/CONFIG_FILE.json
```

For more detailed description of options type:

```text
app/dmss.py -h
```

```text
usage: dmss.py [-h] [-p PORT] [-b BAUD] [-d {8,9}] [-n {1,2,3}] [-s {1,1.5,2}] [-r REFRESH] [-c CONFIG]

dmss.py (Dummy Modbus Slave Server) is configurable modbus slave device simulator.
Register configuration for such fake device, must be provided in form JSON file format.
For examples of such configuration please look int configs/test.json file.

To quit server, please hit CTRL + q key combination!

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  serial port instance to communicate with
  -b BAUD, --baud BAUD  serial port baudrate
  -d {8,9}, --data {8,9}
                        serial port data bits
  -n {1,2,3}, --parity {1,2,3}
                        serial port parity bits.
                        Mapping: 
                        1 -> None
                        2 -> Even
                        3 -> Odd
  -s {1,1.5,2}, --stop {1,1.5,2}
                        serial port stop bits
  -r REFRESH, --refresh REFRESH
                        maximum refresh frequency for registers (1 - 1000Hz)
  -c CONFIG, --config CONFIG
                        MODBUS RTU Device configuration file
```

# TODO
- [x] enable possibility to change values in registers while program is running
- [x] add possiblity to specify refresh rate of values in register for each entity
- [ ] add more comments to code to make it more readible 

# Bugs reporting
If you will find BUG please contact with me ASAP
my e-mail: rafal.tomczyk@likims.pl
