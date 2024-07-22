def init_deepseek_for_deepspeed():
    # This is required to make DeepSpeed to recognize DeepSeek MoE layers
    import deepspeed
    import megatron.model.deepseek.layers as deepseek_layers
    deepspeed.runtime.engine.MoE = deepseek_layers.DeepSeekMoE
    deepspeed.runtime.engine.MoELayer = deepseek_layers.MoELayer
    deepspeed.runtime.engine.TopKGate = deepseek_layers.TopKGate
