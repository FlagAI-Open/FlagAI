
import os 
import torch 

def merge_weight(model_dir):
    model_files = os.listdir(model_dir)
    checkpoint_merge = {}
    print(f"merging the model weight....")
    # multi weights files
    for file_to_load in model_files:
        if "pytorch_model-0" in file_to_load:
            checkpoint_to_load = torch.load(os.path.join(model_dir, file_to_load),map_location="cpu")
            for k, v in checkpoint_to_load.items():
                checkpoint_merge[k] = v
            print(f"{file_to_load} is merged successfully.")
    # save all parameters
    torch.save(
        checkpoint_merge,
        os.path.join(model_dir, "pytorch_model.bin"))
    print(f"models are merged successfully.")


if __name__ == "__main__":
    # merge_weight(model_dir="/share/projset/baaishare/baai-mrnd/xingzhaohu/galactica-6.7b-en/")    
    merge_weight(model_dir="./state_dict/opt-6.7b-en")