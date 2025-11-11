"""
Model Download Sources Module
Provides support for multiple model download sources

This module supports downloading models from:
- BAAI ModelHub (default)
- HuggingFace (including mirrors)
- ModelScope
"""
import os
import json
import requests
from typing import Optional, Dict, List, Tuple
from enum import Enum
import torch
from tqdm.auto import tqdm
from flagai.logger import log_dist


class DownloadSource(Enum):
    """Enumeration of supported download sources."""
    BAAI_MODELHUB = "baai_modelhub"
    HUGGINGFACE = "huggingface"
    HUGGINGFACE_MIRROR = "huggingface_mirror"
    MODEL_SCOPE = "modelscope"


class ModelDownloader:
    """
    Unified model downloader supporting multiple sources.
    
    This class provides a centralized interface for downloading models from
    different sources, including BAAI ModelHub, HuggingFace, HuggingFace mirrors,
    and ModelScope.
    """
    
    def __init__(self, 
                 source: str = "baai_modelhub",
                 hf_mirror: Optional[str] = None,
                 use_auth_token: Optional[str] = None):
        """
        Initialize ModelDownloader.
        
        Args:
            source: Download source ("baai_modelhub", "huggingface", "huggingface_mirror", "modelscope")
            hf_mirror: HuggingFace mirror URL (e.g., "https://hf-mirror.com")
            use_auth_token: HuggingFace authentication token
        """
        self.source = source.lower()
        self.hf_mirror = hf_mirror or os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
        self.use_auth_token = use_auth_token or os.getenv("HF_TOKEN")
        
        # Set default HuggingFace endpoint
        if self.source == "huggingface_mirror":
            self.hf_endpoint = hf_mirror or self.hf_mirror or "https://hf-mirror.com"
        elif self.source == "huggingface":
            self.hf_endpoint = "https://huggingface.co"
        else:
            self.hf_endpoint = None
    
    def download_file(self, 
                     model_name: str,
                     file_name: str,
                     download_path: str,
                     rank: int = 0) -> str:
        """
        Download a file from the configured source.
        
        Args:
            model_name: Name of the model
            file_name: Name of the file to download
            download_path: Directory to save the file
            rank: Process rank for distributed downloads
        
        Returns:
            Path to the downloaded file
        """
        if self.source == "baai_modelhub":
            return self._download_from_baai_modelhub(
                model_name, file_name, download_path, rank
            )
        elif self.source in ["huggingface", "huggingface_mirror"]:
            return self._download_from_huggingface(
                model_name, file_name, download_path, rank
            )
        elif self.source == "modelscope":
            return self._download_from_modelscope(
                model_name, file_name, download_path, rank
            )
        else:
            raise ValueError(f"Unsupported download source: {self.source}")
    
    def _download_from_baai_modelhub(self,
                                     model_name: str,
                                     file_name: str,
                                     download_path: str,
                                     rank: int = 0) -> str:
        """Download from BAAI ModelHub (original implementation)."""
        from flagai.model.file_utils import (
            _get_model_id, _get_checkpoint_path, _get_vocab_path, _get_config_path
        )
        
        try:
            model_id = _get_model_id(model_name)
        except:
            raise ValueError(f"Model {model_name} not found in BAAI ModelHub")
        
        if model_id == "null":
            raise ValueError(f"Model {model_name} not found in BAAI ModelHub")
        
        # Determine file type and use appropriate download function
        if file_name.endswith(".bin") or file_name.endswith(".pth"):
            return _get_checkpoint_path(download_path, file_name, model_id, rank)
        elif file_name.endswith(".json"):
            return _get_config_path(download_path, file_name, model_id, rank)
        else:
            return _get_vocab_path(download_path, file_name, model_id, rank)
    
    def _download_from_huggingface(self,
                                   model_name: str,
                                   file_name: str,
                                   download_path: str,
                                   rank: int = 0) -> str:
        """Download from HuggingFace or HuggingFace mirror."""
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HuggingFace downloads. "
                "Install it with: pip install huggingface_hub"
            )
        
        # Create download path if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        # Determine if we should use mirror
        if self.source == "huggingface_mirror" and self.hf_endpoint:
            # Use mirror endpoint
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        
        try:
            # Download single file
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=file_name,
                cache_dir=download_path,
                token=self.use_auth_token,
                resume_download=True,
            )
            
            # Copy to target location if needed
            target_path = os.path.join(download_path, file_name)
            if file_path != target_path:
                import shutil
                shutil.copy2(file_path, target_path)
                return target_path
            
            return file_path
            
        except Exception as e:
            log_dist(f"Failed to download {file_name} from HuggingFace: {e}", ranks=[0])
            raise
    
    def _download_from_modelscope(self,
                                  model_name: str,
                                  file_name: str,
                                  download_path: str,
                                  rank: int = 0) -> str:
        """Download from ModelScope."""
        try:
            from modelscope.hub.file_download import model_file_download
        except ImportError:
            raise ImportError(
                "modelscope is required for ModelScope downloads. "
                "Install it with: pip install modelscope"
            )
        
        # Create download path if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        try:
            # Download single file
            file_path = model_file_download(
                model_id=model_name,
                file_path=file_name,
                cache_dir=download_path,
            )
            
            # Copy to target location if needed
            target_path = os.path.join(download_path, file_name)
            if file_path != target_path:
                import shutil
                if not os.path.exists(target_path):
                    shutil.copy2(file_path, target_path)
                return target_path
            
            return file_path
            
        except Exception as e:
            log_dist(f"Failed to download {file_name} from ModelScope: {e}", ranks=[0])
            raise
    
    def list_files(self, model_name: str) -> List[str]:
        """
        List available files for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            List of available file names
        """
        if self.source == "baai_modelhub":
            from flagai.model.file_utils import _get_model_files
            try:
                files_str = _get_model_files(model_name)
                return eval(files_str) if files_str else []
            except:
                return []
        elif self.source in ["huggingface", "huggingface_mirror"]:
            try:
                from huggingface_hub import list_repo_files
                if self.source == "huggingface_mirror" and self.hf_endpoint:
                    os.environ["HF_ENDPOINT"] = self.hf_endpoint
                return list_repo_files(
                    repo_id=model_name,
                    token=self.use_auth_token
                )
            except Exception as e:
                log_dist(f"Failed to list files from HuggingFace: {e}", ranks=[0])
                return []
        elif self.source == "modelscope":
            try:
                from modelscope.hub.api import HubApi
                api = HubApi()
                # List files in the model repository
                files_info = api.get_model_files(model_id=model_name)
                if isinstance(files_info, dict) and "Files" in files_info:
                    return [f.get("Name", f.get("Path", "")) for f in files_info["Files"]]
                elif isinstance(files_info, list):
                    return [f.get("Name", f.get("Path", "")) for f in files_info]
                else:
                    return []
            except Exception as e:
                log_dist(f"Failed to list files from ModelScope: {e}", ranks=[0])
                return []
        else:
            return []


def get_downloader(source: Optional[str] = None,
                   hf_mirror: Optional[str] = None,
                   use_auth_token: Optional[str] = None) -> ModelDownloader:
    """
    Get a ModelDownloader instance.
    
    Args:
        source: Download source (defaults to environment variable or "baai_modelhub")
        hf_mirror: HuggingFace mirror URL
        use_auth_token: HuggingFace authentication token
    
    Returns:
        ModelDownloader instance
    """
    if source is None:
        source = os.getenv("FLAGAI_DOWNLOAD_SOURCE", "baai_modelhub")
    
    return ModelDownloader(
        source=source,
        hf_mirror=hf_mirror,
        use_auth_token=use_auth_token
    )


def quick_test_download_source(source: str,
                               model_name: str,
                               file_name: str = "config.json",
                               hf_mirror: Optional[str] = None,
                               use_auth_token: Optional[str] = None) -> bool:
    """
    Quick test for a download source.
    
    Args:
        source: Download source to test
        model_name: Name of the model to test
        file_name: File to test download (default: "config.json")
        hf_mirror: HuggingFace mirror URL
        use_auth_token: HuggingFace authentication token
    
    Returns:
        True if test successful, False otherwise
    
    Examples:
        >>> from flagai.model.download_sources import quick_test_download_source as test_download_source_quick
        >>> # Test HuggingFace mirror
        >>> success = test_download_source_quick(
        ...     source="huggingface_mirror",
        ...     model_name="bert-base-uncased",
        ...     hf_mirror="https://hf-mirror.com"
        ... )
        >>> print(f"Test result: {'PASS' if success else 'FAIL'}")
    """
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp(prefix=f"flagai_test_{source}_")
    
    try:
        downloader = get_downloader(
            source=source,
            hf_mirror=hf_mirror,
            use_auth_token=use_auth_token
        )
        
        file_path = downloader.download_file(
            model_name=model_name,
            file_name=file_name,
            download_path=temp_dir,
            rank=0
        )
        
        # Check if file exists and has content
        if file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"✓ {source}: Successfully downloaded {file_name} from {model_name}")
            return True
        else:
            print(f"✗ {source}: Failed to download {file_name} from {model_name}")
            return False
            
    except Exception as e:
        print(f"✗ {source}: Error - {str(e)}")
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

