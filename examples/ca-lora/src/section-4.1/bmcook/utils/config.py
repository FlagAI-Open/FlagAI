import json

default_values =  { "distillation": {
    "ce_scale": 0,
    "ce_temp": 1,
      
    "mse_hidn_scale": 0,
    "mse_hidn_module": ['[placehold]'],
    "mse_hidn_proj": False,
      
    "mse_att_scale": 0,
    "mse_att_module": ['[placehold]'],
  },

  "pruning": {
    "is_pruning": False,
    "pruning_mask_path": None,
    "pruned_module": ['[placehold]'],
    "mask_method": "m4n2_1d/m4n2_2d/sprune",
    "sprune": {
        "criterion": "l0",
        "training_mask": ['[placehold]'],
        "fixed_mask_path": "",
        "mask_mode": "train_mask",
        "target_sparsity": 0.5
    }
  },

  "quantization": {
    "is_quant": False,
    "quantized_module": [],
  },

  "MoEfication": {
    "is_moefy": False,
    "first_FFN_module": ['[placehold]'],
  }
}

class ConfigParser:

    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def load(self):
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        self.check_default_values()

    def check_default_values(self):
        for key in default_values:
            if key not in self.config:
                self.config[key] = default_values[key]
            else:
                for subkey in default_values[key]:
                    if subkey not in self.config[key]:
                        self.config[key][subkey] = default_values[key][subkey]

    def get(self, key):
        return self.config[key]

    def save(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f)