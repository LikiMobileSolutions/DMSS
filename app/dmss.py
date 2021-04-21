#!/usr/bin/env python3
from argparse import RawTextHelpFormatter, ArgumentParser
import logging
import sys
import time
import threading
import pynput
import os.path
from colorama import Fore, Style

from modbus_device import ModbusDevice
from modbus_server import ModbusServer
from helpers import SerialPortDescriptor

FORMAT = ('%(asctime)-15s %(threadName)-15s %(levelname)-7s %(module)-15s:%(lineno)-4s %(message)s')
logging.basicConfig(format=FORMAT)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

end_program = False

quit_key_combination = [
    {pynput.keyboard.Key.ctrl, pynput.keyboard.KeyCode(char='q')},
    {pynput.keyboard.Key.ctrl_l, pynput.keyboard.KeyCode(char='q')},
    {pynput.keyboard.Key.ctrl_r, pynput.keyboard.KeyCode(char='q')}
]

current = set()


def on_press(key):
    global end_program
    if any([key in comb for comb in quit_key_combination]):
        current.add(key)
        if any(all(k in current for k in comb) for comb in quit_key_combination):
            end_program = True


def on_release(key):
    try:
        current.remove(key)
    except KeyError:
        pass


def main():
    program_description = f"{Fore.YELLOW}dmss.py (Dummy Modbus Slave Server) is configurable modbus slave device simulator.\r\n"
    program_description += "Register configuration for such fake device, must be provided in form JSON file format.\r\n"
    program_description += "For examples of such configuration please look int configs/test.json file.\r\n"
    program_description += "\r\n"
    program_description += f"{Fore.RED}To quit server, please hit CTRL + q key combination!{Style.RESET_ALL}"
    args = ArgumentParser(description=program_description,
                          formatter_class=RawTextHelpFormatter)
    args.add_argument("-p", "--port", required=True,
                      help="serial port instance to communicate with")
    args.add_argument("-b", "--baud", required=False,
                      help="serial port baudrate", type=int, default=9600)
    args.add_argument("-d", "--data", required=False,
                      help="serial port data bits", type=int, default=8, choices=[8, 9])
    args.add_argument("-n", "--parity", required=False, help="serial port parity bits.\r\n" + SerialPortDescriptor.ParityTypes.get_formatted(),
                      type=str, default=SerialPortDescriptor.ParityTypes.NONE, choices=SerialPortDescriptor.ParityTypes.get_values_list())
    args.add_argument("-s", "--stop", required=False,
                      help="serial port stop bits", type=int, default=1, choices=[1, 1.5, 2])
    args.add_argument("-c", "--config", required=False,
                      help="MODBUS RTU Device configuration file", type=str, default="./configs/test.json")
    args_dict = vars(args.parse_args())

    serial = SerialPortDescriptor(port=str(args_dict["port"]), baudrate=int(args_dict["baud"]), datasize=str(
        args_dict["data"]), stopbits=str(args_dict["stop"]), parity=int(args_dict["parity"]))
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, str(args_dict["config"]))
    device = ModbusDevice(path)
    ms = ModbusServer("ModbusThread", device, serial)
    ms.start()

    global end_program
    listener = pynput.keyboard.Listener(
        on_press=on_press, on_release=on_release)
    listener.start()

    while (end_program == False and ms.is_alive()):
        time.sleep(1)

    if ms.is_alive():
        ms.stop()
    listener.stop()


if __name__ == "__main__":
    main()
