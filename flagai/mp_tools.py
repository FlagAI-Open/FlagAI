import sys
import os
import torch
import copy

def check_pytorch_model_mp_size(checkpoint:str, target_mp:int):
    """
    check the checkpoints contains the weights for mp_size = target_mp
    """
    assert os.path.isdir(checkpoint)
    filenames = os.listdir(checkpoint)
    filenames = [
        filename for filename in filenames if filename.startswith("pytorch_model")
    ]
    if 'pytorch_model.bin' in filenames and target_mp==1:
        return True
    else:
        filenames.remove('pytorch_model.bin')
    print("check the weight files in {}, the number of mp_size({}) {} num_of_files({})".format(
        checkpoint, target_mp, "=" if target_mp == len(filenames) else "!=",len(filenames)
    ))
    return target_mp == len(filenames)

def change_pytorch_model_mp_from_1_to_n(checkpoint:str, target_mp:int):
    if check_pytorch_model_mp_size(checkpoint, target_mp):
        return 
    assert os.path.isdir(checkpoint)
    filenames = os.listdir(checkpoint)
    filenames = [
        filename for filename in filenames if filename.startswith("pytorch_model")
    ]
    if 'pytorch_model.bin' in filenames and target_mp>1:
        filenames= ['pytorch_model.bin']
    filenames = [os.path.join(checkpoint, x) for x in filenames]

    if target_mp == len(filenames):
        print("MP size keeps the same.")
        exit(0)

    if checkpoint[-1] == '/':
        new_checkpoint = checkpoint[:-1] 
    else:
        new_checkpoint = checkpoint 
    preserve_keys = [
        "lr_scheduler",
        "skipped_steps",
        "global_steps",
        "global_samples",
        "dp_world_size",
        "iteration",
        "client_lr_scheduler",
        "np_rng_state",
        "random_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
    ]

    if target_mp > len(filenames):
        print("Increase MP size.")
        assert target_mp % len(filenames) == 0
        ratio = target_mp // len(filenames)
        for i in range(len(filenames)):
            start = ratio * i
            end = ratio * (i + 1)
            d = torch.load(filenames[i], map_location='cpu')
            for j in range(start, end):
                d_new = {}
                shift = j - start
                for k, v in d.items():
                    if k != 'module':
                        if k in preserve_keys:
                            d_new[k] = copy.deepcopy(d[k])
                        elif k == "mp_world_size":
                            d_new[k] = target_mp
                        else:
                            d_new[k] = None
                d_new['module'] = {}
                with torch.no_grad():
                    for k, v in d['module'].items():
                        assert len(v.shape) < 3
                        if len(v.shape) == 2 and 'position' not in k:
                            if 'query' in k:
                                part = v.shape[0] // ratio // 3
                                d_new['module'][k] = torch.cat([
                                    v[shift * part:(shift + 1) * part, :].clone(),
                                    v[(shift + ratio) * part:(shift + 1 + ratio) *
                                    part, :].clone(),
                                    v[(shift + 2 * ratio) *
                                    part:(shift + 1 + 2 * ratio) *
                                    part, :].clone()
                                ], 0)
                            elif 'word' in k or 'h_to_4h' in k or 'relative' in k or "r_w_bias" in k or "r_r_bias" in k:
                                part = v.shape[0] // ratio
                                d_new['module'][k] = v[shift * part:(shift + 1) *
                                                    part, :].clone()
                            else:
                                part = v.shape[1] // ratio
                                d_new['module'][k] = v[:,
                                                    shift * part:(shift + 1) *
                                                    part].clone()
                        elif len(v.shape) == 1 and ('dense_h_to_4h' in k
                                                    or "attention.relative" in k):
                            part = v.shape[0] // ratio
                            d_new['module'][k] = v[shift * part:(shift + 1) *
                                                part].clone()
                        elif len(v.shape) == 1 and 'query_key_value' in k:
                            part = v.shape[0] // ratio // 3
                            d_new['module'][k] = torch.cat([
                                v[shift * part:(shift + 1) * part].clone(),
                                v[(shift + ratio) * part:(shift + 1 + ratio) *
                                part].clone(),
                                v[(shift + 2 * ratio) *
                                part:(shift + 1 + 2 * ratio) * part].clone()
                            ], 0)
                        else:
                            d_new['module'][k] = v.clone()
                print("saving mp_size = {:02d} ".format(j))
                filename = os.path.join(new_checkpoint,
                                        "pytorch_model_{:02d}.bin".format(j))
                torch.save(d_new, filename)

if __name__ == "__main__":
    change_pytorch_model_mp_from_1_to_n('/mnt/test_10b_models/state_dict/GLM-10b-en',2)