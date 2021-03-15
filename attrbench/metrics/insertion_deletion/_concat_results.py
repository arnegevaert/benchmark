from typing import List, Dict

import torch


def _concat_results(first: Dict[str, torch.Tensor], middle: Dict[str, List[torch.Tensor]],
                    last: Dict[str, torch.Tensor], baseline: Dict[str, torch.Tensor]):
    result = {}
    for fn in first.keys():
        fn_res = [first[fn]] + middle[fn] + [last[fn]]
        fn_res = torch.cat(fn_res, dim=1)  # [batch_size, len(mask_range)]
        result[fn] = (fn_res / baseline[fn]).cpu()
    return result
