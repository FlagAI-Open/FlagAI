import torch

def head_dimff_prune():
    model = torch.load('/yinxr/gongbt/openBMB/CPM-Live/cpm-live/results/sprune/3B_mask/cpm_live_checkpoint_3B_mask_full-3000.pt')
    layer_zs = torch.load('/yinxr/gongbt/openBMB/CPM-Live/cpm-live/results/sprune/7B/zs.pt')
    att_zs, ffn_zs = layer_zs['self_att_layer_z'], layer_zs['ffn_layer_z']
    head_ff_zs = torch.load('/yinxr/gongbt/openBMB/CPM-Live/cpm-live/results/sprune/3B_mask/ffn_zs.pt')
    heads_z = head_ff_zs['heads_z']  # (37, 32)
    dimff_z = head_ff_zs['dimff_z']

    for k, v in model.items():
        if 'encoder.layers' in k:
            index = int(k.split('.')[2])
            if 'project_' in k:
                if att_zs[index] == 0.:
                    model[k] = v[:2560, :]
                else:
                    heads_mask = heads_z[int(sum(att_zs[:index]))]
                    v = v.view(32, 128, 4096)
                    tgt = []
                    for i, mask in enumerate(heads_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt)
                    assert v.size() == (20, 128, 4096)
                    model[k] = v.view(2560, 4096)
            elif 'attention_out' in k:
                if att_zs[index] == 0.:
                    model[k] = v[:, :2560]
                else:
                    heads_mask = heads_z[int(sum(att_zs[:index]))]
                    v = v.permute(1, 0).view(32, 128, 4096)
                    tgt = []
                    for i, mask in enumerate(heads_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt)
                    assert v.size() == (20, 128, 4096)
                    model[k] = v.view(2560, 4096).permute(1, 0)
            elif 'ffn.ffn.w_in' in k:
                if ffn_zs[index] == 0.:
                    model[k] = v[:3735, :]
                else:
                    dimff_mask = dimff_z[int(sum(ffn_zs[:index]))]
                    assert v.size() == (10240, 4096)
                    tgt = []
                    for i, mask in enumerate(dimff_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt)
                    assert v.size() == (3735, 4096)
                    model[k] = v
            elif 'ffn.ffn.w_out' in k:
                if ffn_zs[index] == 0.:
                    model[k] = v[:, :3735]
                else:
                    dimff_mask = dimff_z[int(sum(ffn_zs[:index]))]
                    v = v.permute(1, 0)
                    assert v.size() == (10240, 4096)
                    tgt = []
                    for i, mask in enumerate(dimff_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt).permute(1, 0)
                    assert v.size() == (4096, 3735)
                    model[k] = v
        if 'position_bias.relative_attention_bias' in k:
            heads_mask = heads_z[0]
            v = v.permute(1, 0).view(32, 1536)
            tgt = []
            for i, mask in enumerate(heads_mask):
                if mask == 1.:
                    tgt.append(v[i])
            v = torch.stack(tgt)
            assert v.size() == (20, 1536)
            model[k] = v.permute(1, 0)

    for k, v in model.items():
        if "attention_out" in k:
            print(k, v.size())

    torch.save(model, '/yinxr/gongbt/openBMB/CPM-Live/cpm-live/results/sprune/3B_mask/cpm_live_checkpoint_3B_mask_pruned.pt')


def layer_prune():
    model = torch.load('/yinxr/cpm_live_ckpt/3B/cpm_live_checkpoint_3B_file2-115500.pt')
    mask = torch.load('/yinxr/cpm_live_ckpt/3B/mask.pt')
    new_model = model.copy()
    att, ffn = 0, 0
    for k in model.keys():
        if 'encoder.layers' in k:
            index = int(k.split('.')[2])
            if mask[index][0] is True:
                    if 'self_att' in k:
                        att += 1
                        del new_model[k]
            if mask[index][1] is True:
                    if 'ffn' in k:
                        ffn += 1
                        del new_model[k]
    torch.save(new_model, '/yinxr/cpm_live_ckpt/3B/cpm_live_checkpoint_3B_pruned.pt')


def mask_update():
    mask_2b = torch.load("results/sprune/2B_layer_mask.pt")
    mask_3b = [[False, False], [False, False], [False, True], [False, True], [False, True], [False, False], [False, True], [False, False], [False, False], [False, False], [False, False], [False, False], [False, False], [True, True], [False, True], [True, False], [False, False], [False, False], [False, False], [True, False], [False, False], [False, True], [False, False], [False, False], [False, False], [True, True], [False, True], [False, False], [True, False], [False, False], [False, True], [False, False], [True, False], [False, True], [True, True], [True, True], [False, False], [False, False], [True, False], [False, False], [False, False], [False, True], [False, False], [True, True], [False, False], [True, False], [False, True], [False, False]]
    att_mask_2b = mask_2b['self_att_layer_z']
    ffn_mask_2b = mask_2b['ffn_layer_z']
    att_index = -1
    ffn_index = -1
    for i, mask in enumerate(mask_3b):
        if mask[0] == False:  # not prune
            att_index += 1
            if att_mask_2b[att_index] == 0: # prune
                mask_3b[i][0] = True
        if mask[1] == False:
            ffn_index += 1
            if ffn_mask_2b[ffn_index] == 0:
                mask_3b[i][1] = True
    print(mask_3b)
    print(len(mask_3b))


def layer_prune_2b():
    model = torch.load('results/sprune/2B_mask/cpm_live_checkpoint_2B_mask-1000.pt')
    new_model = model.copy()
    mask_2b = [[False, False], [False, True], [False, True], [True, True], [False, True], [False, False], [False, True], [False, False], [False, True], [False, False], [True, True], [False, False], [False, True], [True, True], [True, True], [True, True], [False, False], [True, False], [False, False], [True, True], [False, False], [True, True], [True, False], [False, True], [False, False], [True, True], [True, True], [False, False], [True, False], [False, True], [False, True], [False, False], [True, True], [False, True], [True, True], [True, True], [False, False], [True, False], [True, True], [False, False], [False, False], [True, True], [False, True], [True, True], [True, False], [True, False], [True, True], [True, False]]
    att, ffn = 0, 0
    for k in model.keys():
        if 'encoder.layers' in k:
            index = int(k.split('.')[2])
            if mask_2b[index][0] is True:
                    if 'self_att' in k:
                        att += 1
                        del new_model[k]
            if mask_2b[index][1] is True:
                    if 'ffn' in k:
                        ffn += 1
                        del new_model[k]
    torch.save(new_model, 'results/sprune/2B_mask/cpm_live_checkpoint_2B_pruned.pt')
    print("Done")


def head_dimff_prune_1b():
    model = torch.load('results/sprune/300M_mask/cpm_live_checkpoint_300M_mask-83500.pt')
    
    att_zs = [1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
        1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 
        1., 1., 0., 1., 0., 0., 0., 0., 0.]
    ffn_zs = [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 
        1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 
        1., 1., 0., 0., 0., 1., 1., 0., 1.]
    
    head_ff_zs = torch.load('results/sprune/300M_head_mask.pt')
    heads_z = head_ff_zs['heads_z']  # (25, 20)
    dimff_z = head_ff_zs['dimff_z']  # (21, 3735)

    for k, v in model.items():
        if 'encoder.layers' in k:
            index = int(k.split('.')[2])
            if 'project_' in k:
                if att_zs[index] == 0.:
                    print('ERROR!')
                else:
                    att_index = int(sum(att_zs[:index]))
                    heads_mask = heads_z[att_index]
                    v = v.view(16, 128, 4096)
                    tgt = []
                    for i, mask in enumerate(heads_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt)
                    assert v.size() == (4, 128, 4096)
                    model[k] = v.view(512, 4096)
            elif 'attention_out' in k:
                if att_zs[index] == 0.:
                    print('ERROR!')
                else:
                    att_index = int(sum(att_zs[:index]))
                    heads_mask = heads_z[att_index]
                    v = v.permute(1, 0).view(16, 128, 4096)
                    tgt = []
                    for i, mask in enumerate(heads_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt)
                    assert v.size() == (4, 128, 4096)
                    model[k] = v.view(512, 4096).permute(1, 0)
            elif 'ffn.ffn.w_in' in k:
                if ffn_zs[index] == 0.:
                    print('ERROR')
                else:
                    ffn_index = int(sum(ffn_zs[:index]))
                    dimff_mask = dimff_z[ffn_index]
                    assert v.size() == (700, 4096)
                    tgt = []
                    for i, mask in enumerate(dimff_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt)
                    assert v.size() == (350, 4096)
                    model[k] = v
            elif 'ffn.ffn.w_out' in k:
                if ffn_zs[index] == 0.:
                    print('ERROR!')
                else:
                    ffn_index = int(sum(ffn_zs[:index]))
                    dimff_mask = dimff_z[ffn_index]
                    v = v.permute(1, 0)
                    assert v.size() == (700, 4096)
                    tgt = []
                    for i, mask in enumerate(dimff_mask):
                        if mask == 1.:
                            tgt.append(v[i])
                    v = torch.stack(tgt).permute(1, 0)
                    assert v.size() == (4096, 350)
                    model[k] = v
        if 'position_bias.relative_attention_bias' in k:
            heads_mask = heads_z[0]
            v = v.permute(1, 0).view(16, 1536)
            tgt = []
            for i, mask in enumerate(heads_mask):
                if mask == 1.:
                    tgt.append(v[i])
            v = torch.stack(tgt)
            assert v.size() == (4, 1536)
            model[k] = v.permute(1, 0)

    for k, v in model.items():
        if "attention_out" in k:
            print(k, v.size())

    torch.save(model, 'results/sprune/300M_mask/cpm_live_checkpoint_300M_pruned.pt')


if __name__ == "__main__":
    head_dimff_prune_1b()