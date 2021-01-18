import yaml
from . import attribution


class MethodLoader:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def load_config(self, loc):
        with open(loc) as fp:
            methods = {}
            data = yaml.full_load(fp)
            print(data)
            wrappers = []
            if "wrappers" in data.keys():
                # Handle post-processing wrappers
                for wrapper in data["wrappers"].keys():
                    wrappers.append({
                        "constructor": getattr(attribution, wrapper),
                        "args": data["wrappers"][wrapper]
                    })
            if "methods" in data.keys():
                # Handle methods
                for method in data["methods"]:
                    if type(method) == str:
                        constructor = getattr(attribution, method)
                        method_obj = constructor(self.model, **self.kwargs)
                    elif type(method) == dict:
                        if len(method) > 1:
                            raise ValueError("Invalid configuration file")
                        method_id = next(iter(method))
                        constructor = getattr(attribution, method_id)
                        method_obj = constructor(self.model, **self.kwargs, **method[method_id])
                    else:
                        raise ValueError(f"Invalid configuration file")
                    for wrapper in wrappers:
                        method_obj = wrapper["constructor"](method_obj, **wrapper["args"])
                    methods[method] = method_obj
            else:
                raise ValueError(f"Invalid configuration file: file must contain key 'methods'")
            return methods
