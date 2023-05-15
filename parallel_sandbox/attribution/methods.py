from typing import List, Literal
from attrbench.util.attribution_method import AttributionMethod
import torch
from torch import nn
from captum import attr
import saliency.core as saliency
import numpy as np

from torchray.attribution.rise import rise,rise_class
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward, simple_reward

class Gradient(AttributionMethod):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.saliency = attr.Saliency(self.model)

    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return self.saliency.attribute(batch_x, batch_target)


class InputXGradient(AttributionMethod):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.ixg = attr.InputXGradient(self.model)

    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return self.ixg.attribute(batch_x, batch_target)


class IntegratedGradients(AttributionMethod):
    def __init__(self, model: nn.Module, batch_size: int) -> None:
        super().__init__(model)
        self.integrated_gradients = attr.IntegratedGradients(self.model)
        self.batch_size = batch_size

    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return self.integrated_gradients.attribute(
                inputs=batch_x,
                target=batch_target,
                internal_batch_size=self.batch_size)


class Random(AttributionMethod):
    def __call__(self, batch_x: torch.Tensor,
                 batch_target: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(batch_x)


class XRAI(AttributionMethod):
    def __init__(self, model: torch.nn.Module, batch_size: int,**kwargs) -> None:
        super().__init__(model)
        self.batch_size = batch_size
        self.xrai_object = saliency.XRAI()

    
    def _call_model_function_xrai(self,images, call_model_args=None, expected_keys=None):
        with torch.enable_grad():
            class_idx_str = 'class_idx_str'
            device = call_model_args['device']
            images=images.transpose(0,3,1,2)
            images=torch.tensor(images, dtype=torch.float32, device=device)
            images= images.requires_grad_(True)
            target_class_idx =  call_model_args[class_idx_str]
            output = self.model(images)
            # m = torch.nn.Softmax(dim=1)
            # output = m(output)
            assert saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys
            outputs = output[:,target_class_idx]
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            # grads = torch.movedim(grads[0], 1, 3)
            gradients = grads[0].detach().cpu().numpy()
            gradients = gradients.transpose(0,2,3,1)
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        class_idx_str = 'class_idx_str'
        device = batch_x.device
        batch_x=batch_x.detach().cpu().numpy()
        channels=batch_x.shape[1]
        batch_target=batch_target.detach().cpu().numpy()
        res_list= []
        for im, target in zip(batch_x,batch_target):
            call_model_args = {class_idx_str: target, 'device':device}
            xrai_attributions = self.xrai_object.GetMask(im.transpose(1,2,0), self._call_model_function_xrai, call_model_args, batch_size=self.batch_size)
            res_list.append(xrai_attributions)
        result = np.stack(res_list)
        # expand attribtions to have channels for compatibility reasons
        result=np.tile(np.expand_dims(result,axis=1),(1,channels,1,1))
        result=torch.from_numpy(result)
        return result
    

class ExtremalPerturbation(AttributionMethod):
    def __init__(self, model: nn.Module,reward = 'contrastive_reward', areas=[0.1])-> None:
        super().__init__(model)
        if reward=='contrastive_reward':
            self.reward_fuction = contrastive_reward
        else:
            self.reward_fuction = simple_reward
        self.areas=areas

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        attributions=[]
        channels=batch_x.shape[1]
        for x,y in zip(batch_x,batch_target):
            with torch.enable_grad():
                res=extremal_perturbation(self.model,x.unsqueeze(0),int(y),areas=self.areas, reward_func=self.reward_fuction,resize=True)
            mask = res[0]
            attributions.append(mask.detach().cpu())

        result=torch.vstack(attributions)
        result=torch.tile(result,dims=[1,channels,1,1])
        
        return result
    
class Rise(AttributionMethod):
    def __init__(self, model: nn.Module,batch_size=16)-> None:
        super().__init__(model)
        self.batch_size=batch_size

    def __call__(self, batch_x: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        channels=batch_x.shape[1]
        result=rise_class(self.model,input=batch_x,target=batch_target, batch_size=self.batch_size,resize=True)
        result=torch.tile(result,dims=[1,channels,1,1])
        return result
    
        