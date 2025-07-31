from DeltaCenter import OssClient
from .file_utils import default_cache_path


def download(finetuned_delta_path, cache_dir=None, force_download=False):
    if cache_dir is None:
        cache_dir = default_cache_path
    path_to_unzip_file = OssClient.download(finetuned_delta_path, dest=cache_dir, force_download=force_download)
    return path_to_unzip_file

