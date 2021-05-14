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
import time

# import modbus device descriptor helpers
from modbus_device import *
from helpers import SerialPortDescriptor

log = logging.getLogger()


class ModbusServer(threading.Thread):
    def __init__(
        self, thread_name, modbus_device, serial_port_desc=SerialPortDescriptor(), register_refresh_rate=1000
    ):
        threading.Thread.__init__(self, name=thread_name, daemon=True)
        self.__register_refresh_rate = register_refresh_rate
        self.__store = dict()
        self.__device = modbus_device
        self.__serial_desc = serial_port_desc
        self.__identity = ModbusDeviceIdentification()

        self.__identity.VendorName = self.__device.get_vendor()
        self.__identity.ProductCode = self.__device.get_device_code()
        self.__identity.VendorUrl = self.__device.get_vendor_url()
        self.__identity.ProductName = self.__device.get_device_name()
        self.__identity.ModelName = self.__device.get_device_model()
        self.__identity.MajorMinorRevision = self.__device.get_device_revision()

        slaves = self.__device.get_slaves()
        for slave in slaves:
            digital_inputs = ModbusSparseDataBlock(slave.get_inputs())
            coils = ModbusSparseDataBlock(slave.get_coils())
            holding_regs = ModbusSparseDataBlock(slave.get_holding_registers())
            input_regs = ModbusSparseDataBlock(slave.get_in_registers())
            self.__store[slave.get_slave_id()] = ModbusSlaveContext(
                di=digital_inputs, co=coils, ir=input_regs, hr=holding_regs, zero_mode=True
            )
        self.__context = ModbusServerContext(slaves=self.__store, single=False)

        self.__updating_thread = threading.Thread(
            target=self.update_regs, name="UpdatingThread", args=(self.is_alive,), daemon=True
        )

    def run(self):
        try:
            self.__updating_thread.start()
            StartSerialServer(
                context=self.__context,
                identity=self.__identity,
                framer=ModbusRtuFramer,
                port=self.__serial_desc.port,
                baudrate=self.__serial_desc.baudrate,
                bytesize=int(self.__serial_desc.datasize),
                stopbits=int(self.__serial_desc.stopbits),
                parity=SerialPortDescriptor.ParityTypes.ParityTypesDict[self.__serial_desc.parity][0],
                timeout=float(self.__serial_desc.timeout),
                xonoff=int(self.__serial_desc.xonxoff),
                rtscts=int(self.__serial_desc.rtscts),
            )
        except:
            log.error("Wrong serial port configuration specified or serial adapter not connected!")

    def update_regs(self, args):
        is_parent_alive = args
        sleep_time = 1 / self.__register_refresh_rate
        timeout = 10
        t2 = t1 = time.perf_counter()
        while 1:
            if self.is_alive():
                self.__device.update_device_state()
                slaves = self.__device.get_slaves()
                for slave in slaves:
                    slave_id = slave.get_slave_id()
                    self.__context[slave_id].store["d"] = ModbusSparseDataBlock(slave.get_inputs())
                    self.__context[slave_id].store["c"] = ModbusSparseDataBlock(slave.get_coils())
                    self.__context[slave_id].store["h"] = ModbusSparseDataBlock(slave.get_holding_registers())
                    self.__context[slave_id].store["i"] = ModbusSparseDataBlock(slave.get_in_registers())
                t2 = time.perf_counter()
                dT = sleep_time - (t2 - t1)
                if dT < 0.0:
                    raise RuntimeError("Please decrease refresh frequency!")
                time.sleep(dT)
                t1 = time.perf_counter()
            else:
                timeout -= 1
                time.sleep(1)
                if not timeout:
                    break

    def stop(self):
        StopServer()
