from device_profiles.ffle03mq import ModbusProfile
import json
s = ModbusProfile()
s.processDevice()
with open(s.getModel()+".json", "wt") as f:
    json.dump(s.config, f, indent=4)

