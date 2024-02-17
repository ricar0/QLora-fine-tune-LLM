from omegaconf import OmegaConf
import sys
# The default
configs = [OmegaConf.load('config/base_config.yaml')]
configs = OmegaConf.unsafe_merge(*configs)