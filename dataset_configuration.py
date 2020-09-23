from pathlib import Path


class DataSetConfiguration:

    def __init__(self, name: str, channel=1, height=28, width=28, num_classes=10):
        self._name = name.lower()
        self._path = Path('./data')
        self.channel = channel
        self.height = height
        self.width = width
        self.num_classes = num_classes
        for item in ('dataset', 'model', 'log', 'cache'):
            item = item.lower().capitalize()
            path = self._path / item / self.cap_name
            if not path.exists():
                path.mkdir(parents=True)
            self.__dict__[item.lower()] = path

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name.lower()

    @property
    def lower_name(self):
        return self._name.lower()
    
    @property
    def upper_name(self):
        return self._name.upper()

    @property
    def cap_name(self):
        return self._name.capitalize()
