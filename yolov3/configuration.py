import yaml

class ConfigurationModule(object):
	def __init__(self, file: str) -> None:
		super(ConfigurationModule, self).__init__()
		with open(file, mode='r') as fd:
			for key, value in yaml.safe_load(fd).items():
				setattr(self, key, value)

CONFIG = ConfigurationModule('config.yaml')
