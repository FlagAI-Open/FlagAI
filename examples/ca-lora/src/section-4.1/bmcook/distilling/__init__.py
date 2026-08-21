import types
import bmtrain as bmt
import torch.nn.functional as F
import model_center


class BMDistill:
    '''
    BMDistill provide additional training objectives for knowledge distillation, which further improves the performance of compressed models.
    '''

    @classmethod
    def set_forward(cls, student, teacher, foward_fn, config):
        '''
        Modify the forward function of the student model to compute additional knowledge distillation loss.

        `foward_fn` should have the following arguments: `foward_fn(model, enc_input, enc_length, dec_input, dec_length, targets, loss_func)`. These arguments are general for existing Transformers. For decoder-only model, `enc_input` and `enc_length` can be set to None. For encoder-only model, `dec_input` and `dec_length` can be set to None. Similarly, `student` and `teacher` models also have the following arguments: `model(enc_input, enc_length, dec_input, dec_length)`.

        :param student: Student model.
        :param teacher: Teacher model.
        :param foward_fn: Forward function of the student model.
        :param config: ConfigParser object.
        :return: Modified forward function, whose return values are the original return values of `foward_fn` and additional knowledge distillation loss.
        '''

        distill_config = config.get('distillation')
        if distill_config['ce_scale'] + distill_config['mse_hidn_scale'] + distill_config['mse_att_scale'] == 0:
            # if all scales are zero, return the original forward function
            return foward_fn

        if distill_config['mse_hidn_scale'] > 0:
            # if mse_hidn_scale is not zero, get the module mapping from the teacher model to the student model
            s_module_map, t_module_map = get_module_map(distill_config['mse_hidn_module'])
            update_forward(student, teacher, s_module_map, t_module_map)

        if cls.version:
            def forward(model, loss_func, targets, *model_args, **model_kwargs):

                with bmt.inspect.inspect_tensor() as inspector:
                    outputs = foward_fn(
                        model, loss_func, targets, *model_args, **model_kwargs)
                    outputs_t = teacher(*model_args, **model_kwargs)

                records = {}
                for record in inspector._summary:
                    records[record['name']] = record['tensor']

                loss = outputs[0]
                model_outputs = outputs[1]
                logits_s = model_outputs


                # Compute loss and d_loss
                d_loss = 0.0
                if distill_config['ce_scale'] > 0:
                    temp = distill_config['ce_temp']
                    logits_t = outputs_t.detach()
                    prob_t = F.softmax(logits_t / temp, dim=-1)
                    log_prob_s = F.log_softmax(logits_s / temp, dim=-1)
                    d_loss += -(prob_t * log_prob_s).sum(dim=1).mean() * distill_config['ce_scale']
            
                # MSE loss 
                if distill_config['mse_hidn_scale'] > 0:
                    for module_name in s_module_map:
                        t_module_name = s_module_map[module_name]['t']['name']
                        student_t = records[module_name+'_student']
                        teacher_t = records[t_module_name+'_teacher'].detach()

                        if distill_config['mse_hidn_proj']:
                            if 'mapping' not in s_module_map[module_name]:
                                t_dim = teacher_t.size(-1)
                                s_dim = student_t.size(-1)
                                # May be different on different devices
                                
                                s_module_map[module_name]['mapping'] = model_center.layer.Linear(t_dim, s_dim, init_std=0.02)
                                bmt.init_parameters(s_module_map[module_name]['mapping'])
                                s_module_map[module_name]['mapping'].to(teacher_t.device)
                                bmt.synchronize()
                            
                            teacher_t = s_module_map[module_name]['mapping'](teacher_t)
                            
                        cur_loss = (student_t - teacher_t).pow(2).mean() * distill_config['mse_hidn_scale']
                        d_loss += cur_loss
                
                loss = loss + d_loss

                # update loss & append distillation loss
                outputs[0] = loss
                outputs[4] = d_loss
                return outputs
        else:
            def forward(model, loss_func, targets, *model_args, **model_kwargs):

                with bmt.inspect.inspect_tensor() as inspector:
                    outputs = foward_fn(
                        model, loss_func, targets, *model_args, **model_kwargs)    
                    outputs_t = teacher(*model_args, **model_kwargs)


                records = {}
                for record in inspector._summary:
                    records[record['name']] = record['tensor']

                loss = outputs[0]
                model_outputs = outputs[1]
                logits_s = model_outputs

                # Compute loss and d_loss
                d_loss = 0.0
                if distill_config['ce_scale'] > 0:
                    temp = distill_config['ce_temp']
                    logits_t = outputs_t.logits.detach()
                    prob_t = F.softmax(logits_t / temp, dim=-1)
                    log_prob_s = F.log_softmax(logits_s / temp, dim=-1)
                    d_loss += -(prob_t * log_prob_s).sum(dim=1).mean() * distill_config['ce_scale']
            
                # MSE loss 
                if distill_config['mse_hidn_scale'] > 0:
                    for module_name in s_module_map:
                        t_module_name = s_module_map[module_name]['t']['name']
                        student_t = records[module_name+'_student']
                        teacher_t = records[t_module_name+'_teacher'].detach()

                        if distill_config['mse_hidn_proj']:
                            if 'mapping' not in s_module_map[module_name]:
                                t_dim = teacher_t.size(-1)
                                s_dim = student_t.size(-1)
                                # May be different on different devices
                                
                                s_module_map[module_name]['mapping'] = model_center.layer.Linear(t_dim, s_dim, init_std=0.02)
                                bmt.init_parameters(s_module_map[module_name]['mapping'])
                                s_module_map[module_name]['mapping'].to(teacher_t.device)
                                bmt.synchronize()
                            
                            teacher_t = s_module_map[module_name]['mapping'](teacher_t)
                            
                        cur_loss = (student_t - teacher_t).pow(2).mean() * distill_config['mse_hidn_scale']
                        d_loss += cur_loss
                
                loss = loss + d_loss

                # update loss & append distillation loss
                outputs[0] = loss
                outputs[4] = d_loss
                return outputs
        return forward

def get_module_info(info):
    '''
    Parse module info. For example, "[post]encoder.output_layernorm" is parsed to {'name': 'encoder.output_layernorm', 'type': 'post'}, which means the output of the 'encoder.output_layernorm' module is used for distillation. Meanwhile, "[pre]encoder.output_layernorm" is parsed to {'name': 'encoder.output_layernorm', 'type': 'pre'}, which means the input of the 'encoder.output_layernorm' module is used for distillation.

    :param info: Module info.
    '''
    name = info.split(']')[1]
    x_type = info.split(']')[0][1:]
    if x_type in ['post', 'pre']:
        return {"name": name, "type": x_type}
    else:
        raise ValueError('Unknown module type: {}'.format(x_type))

def get_module_map(module_list):
    '''
    Get the module mapping from the teacher model to the student model. For example, "[post]encoder.output_layernorm:[post]encoder.output_layernorm" means that the output of the 'encoder.output_layernorm' module in the teacher model is corresponding to the output of the 'encoder.output_layernorm' module in the student model. The first module name is from the student model, and the second module name is from the teacher model.

    :param module_list: List of module info.
    '''

    s_module_map = {}
    t_module_map = {}
    for pair in module_list:
        s_module, t_module = pair.split(':')
        s_module = get_module_info(s_module)
        t_module = get_module_info(t_module)
        s_module_map[s_module['name']] = {'s': s_module, 't': t_module}
        t_module_map[t_module['name']] = s_module_map[s_module['name']]
    return s_module_map, t_module_map

def update_forward(student, teacher, s_module_map, t_module_map):
    '''
    Update the forward function of target modules in the student and teacher models.

    :param student: Student model.
    :param teacher: Teacher model.
    :param s_module_map: Module mapping from the student model to the teacher model.
    :param t_module_map: Module mapping from the teacher model to the student model.

    '''

    select_keys = set()
    for k, v in student.named_modules():
        if k in s_module_map:
            select_keys.add(k)
            v.forward_old = v.forward
            v.inspect_name = k+'_student'
            
            if s_module_map[k]['s']['type'] == 'pre':
                def _forward(module_self, x):
                    bmt.inspect.record_tensor(x, module_self.inspect_name)
                    return module_self.forward_old(x)
            
            elif s_module_map[k]['s']['type'] == 'post':
                def _forward(module_self, x):
                    x = module_self.forward_old(x)
                    bmt.inspect.record_tensor(x, module_self.inspect_name)
                    return x
            
            v.forward = types.MethodType(_forward, v)

    for k, v in teacher.named_modules():
        if k in t_module_map:
            select_keys.add(k)
            v.forward_old = v.forward
            v.inspect_name = k+'_teacher'

            if t_module_map[k]['t']['type'] == 'pre':
                def _forward(module_self, x):
                    bmt.inspect.record_tensor(x, module_self.inspect_name)
                    return module_self.forward_old(x)

            elif t_module_map[k]['t']['type'] == 'post':
                def _forward(module_self, x):
                    x = module_self.forward_old(x)
                    bmt.inspect.record_tensor(x, module_self.inspect_name)
                    return x
            
            v.forward = types.MethodType(_forward, v)
    bmt.print_rank('Selected modules for hidden state MSE: {}'.format(select_keys))                        
