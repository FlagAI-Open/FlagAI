import types
import model_center
import math
import cpm_kernels.torch as ct

class BMQuant:
    '''
    BMQuant enables quantization-aware training of PLMs by using `cpm-kernels`.
    '''

    @classmethod
    def quantize(cls, model, config):
        '''
        Practitioners can turn on quantization by `is_quant` in the config, which will replace all linear layers with quantized linear layers. BMCook provides the simulation of 8-bit quantization.

        :param model: Model to quantize.
        :param config: Configuration of the quantization.
        '''
        quant_config = config.get('quantization')
        if not quant_config['is_quant']:
            return

        # fix cpm_kernel
        ct.gemm.GEMMInt8._backward = ct.gemm.GEMMInt8.backward
        def new_func(ctx, grad_f):
            if not grad_f.is_contiguous():
                grad_f = grad_f.contiguous()
            return ct.gemm.GEMMInt8._backward(ctx, grad_f)
        ct.gemm.GEMMInt8.backward = new_func

        for name, module in model.named_modules():
            if isinstance(module, model_center.layer.Linear):
                if len(quant_config["quantized_module"]) != 0:
                    if not any([pattern in name for pattern in quant_config["quantized_module"]]):
                        continue
                module.forward = types.MethodType(forward_in8, module)

def forward_in8(module_self, x):
    if module_self.length_scale and module_self.length_scale_before:
        x = x / math.sqrt(module_self.dim_in)
    x = x.transpose(1, 2).contiguous()
    x = ct.bmm(module_self.weight.unsqueeze(0), False, x, False, int8=True)
    x = x.transpose(1, 2).contiguous()
    if module_self.length_scale and not module_self.length_scale_before:
        x = x / math.sqrt(module_self.dim_in)
    if module_self.bias is not None:
        x = x + module_self.bias
    return x
