import json
import random
import struct
from enum import Enum
from helpers import *

# import service logging
import logging

log = logging.getLogger()

to_int = IntConverter()
to_float = FloatConverter()
pretty_print = PrettyDictPrint()
d_sort = SortDict()


class ModbusDeviceIOEntity:
    class EntityType(Enum):
        DISCRETE_INPUT = 0
        COIL_STATE = 1
        HOLDING_REGISTER = 2
        INPUT_REGISTER = 3

    class EntityValueType(Enum):
        BIT = 0
        U16 = 1
        S16 = 2
        U32 = 3
        S32 = 4
        F32 = 5
        STR = 6

    class EntityBehavior(Enum):
        CONST = 0,
        RANDOM = 1,
        PATTERN = 2

    EntityTypeDict = {
        "discreteInputs": EntityType.DISCRETE_INPUT,
        "coilsInitializer": EntityType.COIL_STATE,
        "holdingRegisters": EntityType.HOLDING_REGISTER,
        "inputRegisters": EntityType.INPUT_REGISTER
    }

    EntityValueTypeDict = {
        "BIT": EntityValueType.BIT,
        "U16": EntityValueType.U16,
        "S16": EntityValueType.S16,
        "U32": EntityValueType.U32,
        "S32": EntityValueType.S32,
        "F32": EntityValueType.F32,
        "STR": EntityValueType.STR
    }

    EntityBehaviorDict = {
        "const": EntityBehavior.CONST,
        "random": EntityBehavior.RANDOM,
        "pattern": EntityBehavior.PATTERN
    }

    def __init__(self, __entity_type=str(), slave_description=dict()):
        self.__entity_type = ModbusDeviceIOEntity.EntityTypeDict[__entity_type]
        self.__value_type = ModbusDeviceIOEntity.EntityValueTypeDict[slave_description["type"]]

        if "offset" in slave_description:
            self.__offset = to_int(slave_description["offset"])
            if self.__offset < 1:
                log.error("\"offset\" value cannot be lower than 1!")
                raise RuntimeError()
        else:
            self.__offset = None

        if "bit_position" in slave_description:
            if(self.__value_type != ModbusDeviceIOEntity.EntityValueType.BIT):
                log.error("\"bit_position\" applies only to BIT value type")
                raise RuntimeError()
            self.__bit_position = to_int(
                slave_description["bit_position"])
        else:
            self.__bit_position = None
        if(self.get_offset == None and self.__bit_position == None):
            log.error("Please specify \"offset\" or \"bit_position\" or both!")
            raise RuntimeError()

        self.__behaviour = ModbusDeviceIOEntity.EntityBehaviorDict[slave_description["behaviour"]]
        self.__values = []

        for value in slave_description["value"]:
            self.__values.append(value)

        self.__last_value_idx = 0

        inf = "New entity: offset={offset}, {type}, {value_type}, {behavior}, values={values}, bit position={bit_pos}".format(
            type=self.__entity_type, offset=self.__offset, value_type=self.__value_type, behavior=self.__behaviour, values=self.__values, bit_pos=self.__bit_position)
        log.debug(inf)

    def move_to_next_value(self):
        if(self.__behaviour == ModbusDevice.EntityBehavior.CONST):
            return
        elif(self.__behaviour == ModbusDevice.EntityBehavior.PATTERN):
            self.__last_value_idx = (
                self.__last_value_idx + 1) % (len(self.__values) - 1)
        else:
            self.__values = random.randint(0x00, ModbusSlave.FULL_WORD_MASK)

    def get_entity_type(self):
        return self.__entity_type

    def get_offset(self):
        return self.__offset

    def get_behavior(self):
        return self.__behavior

    def get_value_type(self):
        return self.__value_type

    def get_value(self):
        return self.__values[self.__last_value_idx]

    def get_bit_position(self):
        return self.__bit_position


class ModbusSlave:

    # constants
    WORD_WIDTH = 16
    FULL_WORD_MASK = 0xFFFF
    DEFAULT_OFFSET = 1

    def __init__(self, slave_desc_dict):
        self.__slave_id = int(slave_desc_dict["slaveId"])
        if self.__slave_id < 1:
            log.error("Modbus slave id cannot be smaller than 1!")
            raise RuntimeError()
        discrete_inputs = []
        coils = []
        out_registers = []
        in_registers = []

        for d_in in slave_desc_dict["discreteInputs"]:
            discrete_inputs.append(
                ModbusDeviceIOEntity("discreteInputs", d_in))
        for coil in slave_desc_dict["coilsInitializer"]:
            coils.append(ModbusDeviceIOEntity("coilsInitializer", coil))
        for o_reg in slave_desc_dict["holdingRegisters"]:
            out_registers.append(
                ModbusDeviceIOEntity("holdingRegisters", o_reg))
        for i_reg in slave_desc_dict["inputRegisters"]:
            in_registers.append(
                ModbusDeviceIOEntity("inputRegisters", i_reg))

        self.__discrete_inputs = d_sort(self.__convert_entities_to_dictionary(
            discrete_inputs))
        self.__coils = d_sort(self.__convert_entities_to_dictionary(coils))
        self.__out_registers = d_sort(self.__convert_entities_to_dictionary(
            out_registers))
        self.__in_registers = d_sort(
            self.__convert_entities_to_dictionary(in_registers))

        self.__print_regs_data(self.__discrete_inputs, "discrete inputs")
        self.__print_regs_data(self.__coils, "coils")
        self.__print_regs_data(self.__out_registers, "holding registers")
        self.__print_regs_data(self.__in_registers, "input registers")

    def __print_regs_data(self, reg_dict=dict(), reg_name=str()):
        formatted_data = pretty_print.get_formatted(reg_dict, reg_name)
        for line in formatted_data.splitlines():
            log.debug(line)

    def __check_if_key_exists(key, dictionary=dict()):
        if key in dictionary:
            log.error(
                "Some registers are overlapping. Check your device configuration file!")
            raise RuntimeError()

    def __convert_entities_to_dictionary(self, entities_list=[]):
        converted_values = dict()
        entity_value_type_to_format_specifier = {
            ModbusDeviceIOEntity.EntityValueType.U32.name: '>L',
            ModbusDeviceIOEntity.EntityValueType.S32.name: '>i',
            ModbusDeviceIOEntity.EntityValueType.F32.name: '>f'
        }

        def convert_word_to_bit_dict(start_idx=int(), word=int()):
            BIT_MASK = 0x0001
            bits = dict()
            idx = int(0)
            while idx < ModbusSlave.WORD_WIDTH:
                bits[start_idx + idx] = (word & BIT_MASK)
                word >>= 1
                idx += 1
            return bits

        def convert_bit_dict_to_word_dict(bit_dictionary=dict()):
            words_dict = dict()
            bits_sorted = d_sort(bit_dictionary)
            last_bit_idx = list(bits_sorted.keys())[-1]
            idx = int()
            while idx <= last_bit_idx:
                if idx in bits_sorted:
                    real_ofsset = int(idx / ModbusSlave.WORD_WIDTH)
                    bit_pos = int(idx % ModbusSlave.WORD_WIDTH)
                    bit_value = bits_sorted[idx]
                    temp = int(0)
                    if real_ofsset in words_dict:
                        temp = words_dict[real_ofsset]
                    temp = (temp & (~(1 << bit_pos))) | (bit_value << bit_pos)
                    words_dict[real_ofsset] = temp
                idx += 1
            return words_dict

        def check_for_overlaping_bits(start_idx, end_idx, dict_with_bits):
            idx = start_idx
            while idx < end_idx:
                ModbusSlave.__check_if_key_exists(idx, dict_with_bits)
                idx += 1

        bits_positions_dict = dict()
        for entity in entities_list:
            if(entity.get_value_type() == ModbusDeviceIOEntity.EntityValueType.BIT):
                bit_position = entity.get_bit_position()
                offset = entity.get_offset()
                if (bit_position == None):
                    bit_position = 0
                    log.warning(
                        "Warning! bit_position not specified! 0 is default one!")
                if (offset == None):
                    offset = int(1)
                real_ofsset = int(
                    offset + (bit_position / ModbusSlave.WORD_WIDTH))
                real_bit_position = int(
                    (offset * ModbusSlave.WORD_WIDTH) + bit_position)
                ModbusSlave.__check_if_key_exists(
                    real_bit_position, bits_positions_dict)
                bits_positions_dict[real_bit_position] = int(
                    bool(entity.get_value()))
            elif (int(entity.get_value_type().value) < int(ModbusDeviceIOEntity.EntityValueType.U32.value)):
                key = entity.get_offset()
                word = to_int(entity.get_value()) & ModbusSlave.FULL_WORD_MASK
                begin = key * ModbusSlave.WORD_WIDTH
                end = begin + ModbusSlave.WORD_WIDTH
                bit_dict = convert_word_to_bit_dict(begin, word)
                check_for_overlaping_bits(begin, end, bits_positions_dict)
                bits_positions_dict.update(bit_dict)
            elif (int(entity.get_value_type().value) < int(ModbusDeviceIOEntity.EntityValueType.STR.value)):
                word1 = int()
                word2 = int()
                def convert_to_unsigned(x): return x + 2**32

                if(entity.get_value_type().name != 'F32'):
                    word2, word1 = struct.unpack('>HH', struct.pack(
                        entity_value_type_to_format_specifier[entity.get_value_type().name], convert_to_unsigned(to_int(entity.get_value()))))
                else:
                    word2, word1 = struct.unpack('>HH', struct.pack(
                        '>f', to_float(entity.get_value())))
                begin = int(entity.get_offset() * ModbusSlave.WORD_WIDTH)
                end = (begin + (2 * ModbusSlave.WORD_WIDTH))
                pretty_print = PrettyDictPrint()
                w1_bits = convert_word_to_bit_dict(begin, word1)
                w2_bits = convert_word_to_bit_dict(
                    begin + ModbusSlave.WORD_WIDTH, word2)
                bits_positions_dict.update(w1_bits)
                bits_positions_dict.update(w2_bits)
            else:
                BYTES_IN_WORD = int(2)
                idx = int(0)
                string = str(entity.get_value())
                if len(string) % BYTES_IN_WORD:
                    string += '\x00'
                string = bytearray(string.encode(encoding="UTF-8"))
                mapped_str = map(hex, struct.unpack_from(
                    "H"*(len(string)//BYTES_IN_WORD), string))
                words = []
                for word in mapped_str:
                    words.append(word)
                while(idx < len(words)):
                    key = entity.get_offset() + idx
                    begin = key * ModbusSlave.WORD_WIDTH
                    end = begin + ModbusSlave.WORD_WIDTH
                    check_for_overlaping_bits(begin, end, bits_positions_dict)
                    bit_dict = convert_word_to_bit_dict(
                        begin, to_int(words[idx]))
                    bits_positions_dict.update(bit_dict)
                    idx = idx + 1
        converted_values = convert_bit_dict_to_word_dict(bits_positions_dict)
        return converted_values

    def get_slave_id(self):
        return self.__slave_id

    def get_coils(self):
        return self.__coils

    def get_inputs(self):
        return self.__discrete_inputs

    def get_holding_registers(self):
        return self.__out_registers

    def get_in_registers(self):
        return self.__in_registers


class ModbusDevice:
    def __init__(self, device_config_file=dict()):
        data = None

        with open(device_config_file) as f:
            data = json.load(f)

        data = data["device"][0]

        self.__vendor = data["vendorName"]
        self.__vendor_url = data["vendorURL"]
        self.__device_name = data["deviceName"]
        self.__device_model = data["deviceModel"]
        self.__device_code = data["deviceCode"]
        self.__device_rev = data["deviceRevision"]
        self.__slaves = []
        for slave in data["slaves"]:
            self.__slaves.append(ModbusSlave(slave))
        log.debug("###################################################")
        log.debug("New device parsed!")
        log.debug("Device vendor:\t" + self.__vendor)
        log.debug("Vendor website:\t" + self.__vendor_url)
        log.debug("Device name:\t" + self.__device_name)
        log.debug("Device model:\t" + self.__device_model)
        log.debug("Device code:\t" + self.__device_code)
        log.debug("Device revision:\t" + self.__device_rev)
        log.debug("Number of slaves:\t" + str(len(self.__slaves)))
        log.debug("###################################################")
        log.debug("")

    def get_vendor(self):
        return self.__vendor

    def get_vendor_url(self):
        return self.__vendor_url

    def get_device_name(self):
        return self.__device_name

    def get_device_model(self):
        return self.__device_model

    def get_device_code(self):
        return self.__device_code

    def get_device_revision(self):
        return self.__device_rev

    def get_slaves(self):
        return self.__slaves
