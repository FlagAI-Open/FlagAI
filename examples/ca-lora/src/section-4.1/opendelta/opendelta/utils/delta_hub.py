

def create_hub_repo_name(root = "DeltaHub",
                         dataset = None,
                         delta_type = None,
                         model_name_or_path = None,
                         center_value_only_tags = None,
                         center_key_value_tags = None
                         ):
    r"""Currently, it's only a simple concatenation of the arguments. 
    """
    repo_name = []

    repo_name.append(f"{delta_type}")
    model_name_or_path = model_name_or_path.split("/")[-1]
    repo_name.append(f"{model_name_or_path}")
    repo_name.append(f"{dataset}")

    repo_name.extend(list(center_value_only_tags) if center_value_only_tags else [None])
    repo_name.extend([f"{k}-{v}" for k,v in center_key_value_tags.items()] if center_key_value_tags else [None])

    repo_name = "_".join(repo_name)

    repo_name = root+"/"+repo_name
    return repo_name




