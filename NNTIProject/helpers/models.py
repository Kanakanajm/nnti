
from functools import partial
from helpers.general import Cacheable
from transformers import AutoModelForCausalLM

from helpers.lora import LinearWithLoRA

class ModelPreparer(Cacheable):
    def __init__(self, model_name, cache_dir="cache/") -> None:
        super().__init__(cache_dir)
        self.model_name = model_name

    def prepare(self, model):
        pass

    def generate(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir_models).to("cuda")
        self.prepare(model)
        return model

    @property
    def cache_dir_models(self):
        return self.cache_dir_sub("models")
    

class XGLMModelPreparer(ModelPreparer):
    def __init__(self, model_name="facebook/xglm-564M") -> None:
        super().__init__(model_name)

class XGLMBitFitModelPreparer(XGLMModelPreparer):
    def __init__(self) -> None:
        super().__init__()
    def prepare(self, model):
        # bias term fine tuning
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False

class XGLMLoRAModelPreparer(XGLMModelPreparer):
    def __init__(self, r = 8, a = 16) -> None:
        self.r = r
        self.a = a
        super().__init__()
    def prepare(self, model):
        # low-rank adaptation
        assign_lora = partial(LinearWithLoRA, r=self.r, a=self.a)
        for param in model.parameters():
            param.requires_grad = False
    
        for layer in model.model.layers:
            # query
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            # value
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
