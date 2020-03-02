# Singleton metaclass allows for efficient and clean singleton pattern
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Arguments is a singleton dictionary
# Its value is set in the main script by parsing arguments
# Creating an instance anywhere in the project of Arguments gives you the same argument values
class Arguments(metaclass=Singleton):
    def __init__(self, args):
        self.args = args

    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, value):
        raise Exception("Cannot change argument values after parsing")
