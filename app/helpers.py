import struct
from ctypes import *
import logging
from enum import Enum

log = logging.getLogger()


class IntConverter(object):
    def __call__(self, data):
        ret_value = 0
        if type(int()) == type(data):
            ret_value = data
        else:
            bases = (10, 2, 16)
            for base in bases:
                try:
                    ret_value = int(data, base=base)
                except:
                    continue
                break
        return int(ret_value)

    @staticmethod
    def convert_to_unsigned16(x):
        return x + 2 ** 16

    @staticmethod
    def convert_to_unsigned32(x):
        return x + 2 ** 32


class FloatConverter(object):
    def __call__(self, data):
        ret_value = None
        if type(float()) == type(data):
            ret_value = data
        else:
            int_conv = IntConverter()
            temp = int_conv(data)
            cp = pointer(c_int(temp))
            fp = cast(cp, POINTER(c_float))
            ret_value = fp.contents.value
        return float(ret_value)


class PrettyDictPrint(object):
    def __call__(self, data=dict(), name=str()):
        print(self.get_formatted(data, name))

    def get_formatted(self, data=dict(), name=str()):
        formated_str = name + " = {\r\n"
        for offset, value in data.items():
            formated_str += str("  {offset} : {value}\r\n".format(offset=hex(offset), value=hex(value)))
        formated_str += "}"
        return formated_str


class SortDict(object):
    def __call__(self, data=dict()):
        return dict(sorted(data.items()))


class SerialPortDescriptor:
    class ParityTypes:
        NONE, EVEN, ODD = range(1, 4)
        ParityTypesDict = {
            NONE: "None",
            EVEN: "Even",
            ODD: "Odd",
        }

        @staticmethod
        def get_names_list():
            return [ParityTypesDict[NONE], ParityTypesDict[EVEN], ParityTypesDict[ODD]]

        @staticmethod
        def get_values_list():
            return [
                SerialPortDescriptor.ParityTypes.NONE,
                SerialPortDescriptor.ParityTypes.EVEN,
                SerialPortDescriptor.ParityTypes.ODD,
            ]

        @staticmethod
        def get_formatted():
            ret_val = "Mapping: \n"
            for key, value in SerialPortDescriptor.ParityTypes.ParityTypesDict.items():
                ret_val += str("{key} -> {value}\n".format(key=key, value=value))
            return ret_val

    def __init__(
        self,
        port="/dev/ttyS0",
        baudrate=9600,
        datasize=8,
        stopbits=1,
        parity=ParityTypes.NONE,
        timeout=0.1,
        xonxoff=0,
        rtscts=0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.datasize = datasize
        self.stopbits = stopbits
        self.parity = parity
        self.timeout = timeout
        self.xonxoff = xonxoff
        self.rtscts = rtscts
