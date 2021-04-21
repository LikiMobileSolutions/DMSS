# import needed pyModbus API
from pymodbus.server.asynchronous import StartSerialServer, StopServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSparseDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.transaction import ModbusRtuFramer
# import service logging
import logging
# import threading module
import threading
# import modbus device descriptor helpers
from modbus_device import *
from helpers import SerialPortDescriptor
log = logging.getLogger()


class ModbusServer(threading.Thread):
    def __init__(self, thread_name, modbus_device, serial_port_desc=SerialPortDescriptor()):
        threading.Thread.__init__(self, name=thread_name, daemon=True)
        self.__device = modbus_device
        self.__serial_desc = serial_port_desc
        self.__context = None
        self.__identity = ModbusDeviceIdentification()

        self.__identity.VendorName = self.__device.get_vendor()
        self.__identity.ProductCode = self.__device.get_device_code()
        self.__identity.VendorUrl = self.__device.get_vendor_url()
        self.__identity.ProductName = self.__device.get_device_name()
        self.__identity.ModelName = self.__device.get_device_model()
        self.__identity.MajorMinorRevision = self.__device.get_device_revision()

        store = dict()
        for slave in self.__device.get_slaves():
            digital_inputs = ModbusSparseDataBlock(slave.get_inputs())
            coils = ModbusSparseDataBlock(slave.get_coils())
            holding_regs = ModbusSparseDataBlock(slave.get_holding_registers())
            input_regs = ModbusSparseDataBlock(slave.get_holding_registers())
            store[slave.get_slave_id()] = ModbusSlaveContext(
                di=digital_inputs, co=coils, ir=input_regs, hr=holding_regs)

        self.__context = ModbusServerContext(slaves=store, single=False)

    def run(self):
        try:
            StartSerialServer(context=self.__context, identity=self.__identity,
                              framer=ModbusRtuFramer, port=self.__serial_desc.port,
                              baudrate=self.__serial_desc.baudrate,
                              bytesize=int(self.__serial_desc.datasize),
                              stopbits=int(self.__serial_desc.stopbits),
                              parity=SerialPortDescriptor.ParityTypes.ParityTypesDict[
                                  self.__serial_desc.parity][0],
                              timeout=float(self.__serial_desc.timeout),
                              xonoff=int(self.__serial_desc.xonxoff),
                              rtscts=int(self.__serial_desc.rtscts))
        except:
            log.error(
                "Wrong serial port configuration specified or serial adapter not connected!")

    def stop(self):
        StopServer()
