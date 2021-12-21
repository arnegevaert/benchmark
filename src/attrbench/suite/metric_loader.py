import yaml
import inspect
from os import path
from typing import Dict, Callable
from attrbench.metrics import Metric
from attrbench import metrics



class MetricLoader:
    def __init__(self, filename: str, model: Callable, methods: Dict[str, Callable],
                 log_dir: str = None, **kwargs):
        self.filename = filename
        self.global_args = {
            "model": model,
            "methods": methods,
            "method_names": list(methods.keys())
        }
        for arg in kwargs:
            self.global_args[arg] = kwargs[arg]
        self.log_dir = log_dir

    def _parse_section(self, section: Dict, section_name: str, prefix: str = None, section_args: Dict = None) -> Dict[str, Metric]:
        # Only keywords "metrics", "default" and "foreach" are allowed in section root
        for key in section.keys():
            if key not in ("metrics", "default"):
                raise ValueError(f"Invalid configuration file: illegal key {key} in section {section_name}")
        # Parse section default arguments
        default_args = {**self.global_args, **section.get("default", {})}
        if section_args is not None:
            default_args = {**default_args, **section_args}
        # Parse section metrics
        result = {}
        for metric_name in section["metrics"]:
            m_dict = section["metrics"][metric_name]
            # Add prefix if necessary
            if prefix is not None:
                metric_name = f"{prefix}.{metric_name}"
            # Get metric constructor
            constructor = getattr(metrics, m_dict["type"])
            # metric_args contains specific args for this single metric
            metric_args = {key: m_dict[key] for key in m_dict if key != "type"}
            if self.log_dir is not None:
                metric_args["writer_dir"] = path.join(self.log_dir, metric_name)
            # Fill in all expected args (default_args may contain args that are not applicable to this metric)
            signature = inspect.signature(constructor).parameters
            # all_args contains union(default_args, section_args, metric_args)
            all_args = {**default_args, **metric_args}
            # args contains the arguments in all_args that are applicable to this metric
            args = {key: all_args[key] for key in all_args if key in signature}
            # Check if all necessary arguments are present
            for arg in signature:
                if signature[arg].default == inspect.Parameter.empty and (arg not in args or args[arg] is None):
                    raise ValueError(f"Invalid configuration: required argument {arg} "
                                     f"not found for metric {metric_name}")
            # Construct metric
            result[metric_name] = constructor(**args)
        return result

    def load(self) -> Dict[str, Metric]:
        with open(self.filename) as fp:
            data = yaml.full_load(fp)
            if "metrics" in data.keys():
                # If root contains a key "metrics", there are no subsections
                return self._parse_section(data, section_name="root")
            else:
                # Otherwise, parse each subsection
                result = {}
                for section in data.keys():
                    # Section can't have keyword as name
                    if section in ("metrics", "default"):
                        raise ValueError(f"Invalid configuration file: illegal section '{section}' in root")
                    result = {**result, **self._parse_section(data[section], section_name=section)}
                return result
