# Copyright © 2024 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""
Unit tests for download_sources module

Download strategy:
- Small files (< 1GB): tokenizer files, config files, etc. - allow real download
- Large files (>= 1GB): model checkpoints, weights - use mock to avoid long download times
"""
import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from flagai.model.download_sources import (
    ModelDownloader,
    DownloadSource,
    get_downloader
)
# Import quick_test_download_source separately to avoid pytest treating it as a fixture
try:
    from flagai.model.download_sources import quick_test_download_source as test_download_source_quick
except ImportError:
    test_download_source_quick = None

# File types that are typically small and can be downloaded for real testing
SMALL_FILE_EXTENSIONS = {'.json', '.txt', '.jsonl', '.yaml', '.yml', '.toml', '.md'}
SMALL_FILE_PATTERNS = ['tokenizer', 'vocab', 'config', 'merges', 'special_tokens', 'tokenizer_config']

# File types that are typically large and should be mocked
LARGE_FILE_EXTENSIONS = {'.bin', '.pth', '.pt', '.safetensors', '.ckpt', '.h5', '.pb'}

# Size threshold for mocking (1GB)
MOCK_SIZE_THRESHOLD = 1024 * 1024 * 1024  # 1GB in bytes

def should_mock_download(file_name: str, file_size: int = None) -> bool:
    """
    Determine if a file download should be mocked.
    
    Args:
        file_name: Name of the file to download
        file_size: Size of the file in bytes (if known)
    
    Returns:
        True if download should be mocked, False if real download is allowed
    """
    # If file size is known and >= 1GB, mock it
    if file_size is not None and file_size >= MOCK_SIZE_THRESHOLD:
        return True
    
    # Check file extension
    file_ext = os.path.splitext(file_name)[1].lower()
    if file_ext in LARGE_FILE_EXTENSIONS:
        return True
    
    # Small files (config, tokenizer, etc.) can be downloaded
    if file_ext in SMALL_FILE_EXTENSIONS:
        return False
    
    # Check filename patterns
    file_name_lower = file_name.lower()
    for pattern in SMALL_FILE_PATTERNS:
        if pattern in file_name_lower:
            return False
    
    # Default: mock for safety (unknown file types)
    return True


class TestModelDownloader(unittest.TestCase):
    """Test cases for ModelDownloader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_name = "test-model"
        self.file_name = "config.json"

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_model_downloader_initialization_default(self):
        """Test ModelDownloader initialization with default source"""
        downloader = ModelDownloader()
        self.assertEqual(downloader.source, "baai_modelhub")

    def test_model_downloader_initialization_custom_source(self):
        """Test ModelDownloader initialization with custom source"""
        downloader = ModelDownloader(source="huggingface")
        self.assertEqual(downloader.source, "huggingface")

    def test_model_downloader_initialization_huggingface_mirror(self):
        """Test ModelDownloader initialization with HuggingFace mirror"""
        with patch.dict(os.environ, {}, clear=True):
            downloader = ModelDownloader(
                source="huggingface_mirror",
                hf_mirror="https://hf-mirror.com"
            )
            self.assertEqual(downloader.source, "huggingface_mirror")
            self.assertEqual(downloader.hf_endpoint, "https://hf-mirror.com")

    @patch.dict(os.environ, {'HF_ENDPOINT': 'https://custom-mirror.com'})
    def test_model_downloader_initialization_from_env(self):
        """Test ModelDownloader initialization from environment variables"""
        downloader = ModelDownloader(source="huggingface_mirror")
        self.assertEqual(downloader.hf_endpoint, "https://custom-mirror.com")

    def test_model_downloader_unsupported_source(self):
        """Test ModelDownloader with unsupported source"""
        downloader = ModelDownloader(source="unsupported_source")
        with self.assertRaises(ValueError):
            downloader.download_file(
                self.model_name, self.file_name, self.temp_dir
            )

    @patch('flagai.model.file_utils._get_model_id')
    @patch('flagai.model.file_utils._get_checkpoint_path')
    def test_model_downloader_baai_modelhub_checkpoint(self, mock_get_checkpoint, mock_get_id):
        """Test ModelDownloader download from BAAI ModelHub (checkpoint) - Mocked (large file)"""
        # model.bin is a large file, so we mock it
        mock_get_id.return_value = "test_model_id"
        mock_file_path = os.path.join(self.temp_dir, "model.bin")
        mock_get_checkpoint.return_value = mock_file_path
        
        # Create a mock file to simulate downloaded file
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(mock_file_path, 'w') as f:
            f.write("mock checkpoint data")
        
        downloader = ModelDownloader(source="baai_modelhub")
        result = downloader.download_file(
            self.model_name, "model.bin", self.temp_dir
        )
        self.assertIsNotNone(result)
        self.assertEqual(result, mock_file_path)
        mock_get_id.assert_called_once()
        mock_get_checkpoint.assert_called_once()

    @patch('flagai.model.file_utils._get_model_id')
    @patch('flagai.model.file_utils._get_config_path')
    def test_model_downloader_baai_modelhub_config(self, mock_get_config, mock_get_id):
        """Test ModelDownloader download from BAAI ModelHub (config) - Can use real download for small files"""
        # config.json is a small file, but we still mock it for unit test isolation
        # In integration tests, this could be a real download
        mock_get_id.return_value = "test_model_id"
        mock_file_path = os.path.join(self.temp_dir, "config.json")
        mock_get_config.return_value = mock_file_path
        
        # Create a mock file to simulate downloaded file
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(mock_file_path, 'w') as f:
            f.write('{"mock": "config"}')
        
        downloader = ModelDownloader(source="baai_modelhub")
        result = downloader.download_file(
            self.model_name, "config.json", self.temp_dir
        )
        self.assertIsNotNone(result)
        self.assertEqual(result, mock_file_path)
        mock_get_config.assert_called_once()
        
        # Verify that config.json should not be mocked (small file)
        self.assertFalse(should_mock_download("config.json"))

    @patch('huggingface_hub.hf_hub_download')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    def test_model_downloader_huggingface_small_file(self, mock_copy2, mock_makedirs, mock_hf_download):
        """Test ModelDownloader download from HuggingFace (small file) - Can use real download"""
        # config.json is a small file, but we mock it for unit test isolation
        mock_file_path = os.path.join(self.temp_dir, "config.json")
        mock_hf_download.return_value = mock_file_path
        
        # Create a mock file to simulate downloaded file
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(mock_file_path, 'w') as f:
            f.write('{"mock": "config"}')
        
        downloader = ModelDownloader(source="huggingface")
        result = downloader.download_file(
            self.model_name, "config.json", self.temp_dir
        )
        self.assertIsNotNone(result)
        mock_hf_download.assert_called_once()
        # Verify correct parameters were passed
        call_kwargs = mock_hf_download.call_args[1]
        self.assertEqual(call_kwargs['repo_id'], self.model_name)
        self.assertEqual(call_kwargs['filename'], "config.json")
        
        # Verify that config.json should not be mocked (small file)
        self.assertFalse(should_mock_download("config.json"))

    @patch('huggingface_hub.hf_hub_download')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    def test_model_downloader_huggingface_large_file(self, mock_copy2, mock_makedirs, mock_hf_download):
        """Test ModelDownloader download from HuggingFace (large file) - Must be mocked"""
        # model.safetensors is a large file, must be mocked
        mock_file_path = os.path.join(self.temp_dir, "model.safetensors")
        mock_hf_download.return_value = mock_file_path
        
        downloader = ModelDownloader(source="huggingface")
        result = downloader.download_file(
            self.model_name, "model.safetensors", self.temp_dir
        )
        self.assertIsNotNone(result)
        mock_hf_download.assert_called_once()
        
        # Verify that model.safetensors should be mocked (large file)
        self.assertTrue(should_mock_download("model.safetensors"))

    @patch('huggingface_hub.hf_hub_download')
    @patch('os.makedirs')
    @patch.dict(os.environ, {}, clear=True)
    def test_model_downloader_huggingface_mirror(self, mock_makedirs, mock_hf_download):
        """Test ModelDownloader download from HuggingFace mirror - Can use real download for small files"""
        mock_file_path = os.path.join(self.temp_dir, "config.json")
        mock_hf_download.return_value = mock_file_path
        
        # Create a mock file to simulate downloaded file
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(mock_file_path, 'w') as f:
            f.write('{"mock": "config"}')
        
        # Clear environment and test
        with patch.dict(os.environ, {}, clear=True):
            downloader = ModelDownloader(
                source="huggingface_mirror",
                hf_mirror="https://hf-mirror.com"
            )
            result = downloader.download_file(
                self.model_name, "config.json", self.temp_dir
            )
            self.assertIsNotNone(result)
            # Should set HF_ENDPOINT
            self.assertEqual(os.environ.get('HF_ENDPOINT'), "https://hf-mirror.com")
            mock_hf_download.assert_called_once()
        
        # Verify that config.json should not be mocked (small file)
        self.assertFalse(should_mock_download("config.json"))

    @patch('modelscope.hub.file_download.model_file_download')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    def test_model_downloader_modelscope_small_file(self, mock_copy2, mock_makedirs, mock_model_download):
        """Test ModelDownloader download from ModelScope (small file) - Can use real download"""
        mock_file_path = os.path.join(self.temp_dir, "config.json")
        mock_model_download.return_value = mock_file_path
        
        # Create a mock file to simulate downloaded file
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(mock_file_path, 'w') as f:
            f.write('{"mock": "config"}')
        
        downloader = ModelDownloader(source="modelscope")
        result = downloader.download_file(
            self.model_name, "config.json", self.temp_dir
        )
        self.assertIsNotNone(result)
        mock_model_download.assert_called_once()
        # Verify correct parameters were passed
        call_kwargs = mock_model_download.call_args[1]
        self.assertEqual(call_kwargs['model_id'], self.model_name)
        self.assertEqual(call_kwargs['file_path'], "config.json")
        
        # Verify that config.json should not be mocked (small file)
        self.assertFalse(should_mock_download("config.json"))

    @patch('builtins.__import__')
    @patch('os.makedirs')
    @patch('shutil.copy2')
    def test_model_downloader_modelscope_large_file(self, mock_makedirs, mock_copy2, mock_import):
        """Test ModelDownloader download from ModelScope (large file) - Must be mocked"""
        # Mock the import to avoid ModuleNotFoundError
        mock_modelscope = MagicMock()
        mock_file_download = MagicMock()
        mock_file_path = os.path.join(self.temp_dir, "model.bin")
        mock_file_download.model_file_download.return_value = mock_file_path
        mock_modelscope.hub.file_download = mock_file_download
        
        def side_effect(name, *args, **kwargs):
            if name == 'modelscope':
                return mock_modelscope
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        # model.bin is a large file, must be mocked
        downloader = ModelDownloader(source="modelscope")
        result = downloader.download_file(
            self.model_name, "model.bin", self.temp_dir
        )
        self.assertIsNotNone(result)
        mock_file_download.model_file_download.assert_called_once()
        
        # Verify that model.bin should be mocked (large file)
        self.assertTrue(should_mock_download("model.bin"))

    @patch('huggingface_hub.hf_hub_download')
    @patch('os.makedirs')
    def test_model_downloader_huggingface_with_auth_token(self, mock_makedirs, mock_hf_download):
        """Test ModelDownloader with authentication token - Can use real download for small files"""
        mock_file_path = os.path.join(self.temp_dir, "config.json")
        mock_hf_download.return_value = mock_file_path
        
        # Create a mock file to simulate downloaded file
        os.makedirs(self.temp_dir, exist_ok=True)
        with open(mock_file_path, 'w') as f:
            f.write('{"mock": "config"}')
        
        downloader = ModelDownloader(
            source="huggingface",
            use_auth_token="test_token"
        )
        result = downloader.download_file(
            self.model_name, "config.json", self.temp_dir
        )
        self.assertIsNotNone(result)
        # Should pass token to hf_hub_download
        call_kwargs = mock_hf_download.call_args[1]
        self.assertEqual(call_kwargs.get('token'), "test_token")
        
        # Verify that config.json should not be mocked (small file)
        self.assertFalse(should_mock_download("config.json"))

    @patch('builtins.__import__')
    def test_model_downloader_huggingface_import_error(self, mock_import):
        """Test ModelDownloader when huggingface_hub is not installed - Mocked"""
        # Mock the import to raise ImportError
        def side_effect(name, *args, **kwargs):
            if name == 'huggingface_hub':
                raise ImportError("huggingface_hub not found")
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        downloader = ModelDownloader(source="huggingface")
        with self.assertRaises(ImportError):
            downloader.download_file(
                self.model_name, self.file_name, self.temp_dir
            )

    @patch('flagai.model.file_utils._get_model_files')
    def test_model_downloader_list_files_baai_modelhub(self, mock_get_model_files):
        """Test ModelDownloader list_files for BAAI ModelHub - Mocked"""
        # Mock the file list response
        mock_get_model_files.return_value = '["config.json", "model.bin", "vocab.txt"]'
        
        downloader = ModelDownloader(source="baai_modelhub")
        files = downloader.list_files(self.model_name)
        # Should return a list
        self.assertIsInstance(files, list)
        self.assertEqual(len(files), 3)
        self.assertIn("config.json", files)
        mock_get_model_files.assert_called_once_with(self.model_name)

    @patch('huggingface_hub.list_repo_files')
    def test_model_downloader_list_files_huggingface(self, mock_list_files):
        """Test ModelDownloader list_files for HuggingFace - Mocked"""
        mock_list_files.return_value = ["config.json", "model.bin", "vocab.txt"]
        
        downloader = ModelDownloader(source="huggingface")
        files = downloader.list_files(self.model_name)
        # Should return a list
        self.assertIsInstance(files, list)
        self.assertEqual(len(files), 3)
        mock_list_files.assert_called_once_with(
            repo_id=self.model_name,
            token=None
        )

    @patch('huggingface_hub.list_repo_files')
    def test_model_downloader_list_files_huggingface_mirror(self, mock_list_files):
        """Test ModelDownloader list_files for HuggingFace mirror - Mocked"""
        mock_list_files.return_value = ["config.json", "model.bin"]
        
        with patch.dict(os.environ, {}, clear=True):
            downloader = ModelDownloader(
                source="huggingface_mirror",
                hf_mirror="https://hf-mirror.com"
            )
            files = downloader.list_files(self.model_name)
            # Should return a list
            self.assertIsInstance(files, list)
            # Should set HF_ENDPOINT
            self.assertEqual(os.environ.get('HF_ENDPOINT'), "https://hf-mirror.com")
            mock_list_files.assert_called_once()

    @patch('builtins.__import__')
    def test_model_downloader_list_files_modelscope(self, mock_import):
        """Test ModelDownloader list_files for ModelScope - Mocked"""
        # Mock the import to avoid ModuleNotFoundError
        mock_modelscope = MagicMock()
        mock_hub_api = MagicMock()
        mock_api = MagicMock()
        mock_api.get_model_files.return_value = {
            "Files": [
                {"Name": "config.json"},
                {"Name": "model.bin"},
                {"Path": "vocab.txt"}
            ]
        }
        mock_hub_api.HubApi.return_value = mock_api
        mock_modelscope.hub.api = mock_hub_api
        
        def side_effect(name, *args, **kwargs):
            if name == 'modelscope':
                return mock_modelscope
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        downloader = ModelDownloader(source="modelscope")
        files = downloader.list_files(self.model_name)
        # Should return a list
        self.assertIsInstance(files, list)
        self.assertEqual(len(files), 3)
        mock_api.get_model_files.assert_called_once_with(model_id=self.model_name)


class TestDownloadStrategy(unittest.TestCase):
    """Test cases for download strategy (mock vs real download)"""

    def test_should_mock_download_small_files(self):
        """Test that small files should not be mocked"""
        self.assertFalse(should_mock_download("config.json"))
        self.assertFalse(should_mock_download("tokenizer_config.json"))
        self.assertFalse(should_mock_download("vocab.txt"))
        self.assertFalse(should_mock_download("merges.txt"))
        self.assertFalse(should_mock_download("special_tokens_map.json"))
        self.assertFalse(should_mock_download("tokenizer.json"))

    def test_should_mock_download_large_files(self):
        """Test that large files should be mocked"""
        self.assertTrue(should_mock_download("model.bin"))
        self.assertTrue(should_mock_download("model.safetensors"))
        self.assertTrue(should_mock_download("pytorch_model.bin"))
        self.assertTrue(should_mock_download("model.pt"))
        self.assertTrue(should_mock_download("checkpoint.pth"))

    def test_should_mock_download_by_size(self):
        """Test that files >= 1GB should be mocked"""
        # Small file (< 1GB) - should not mock
        self.assertFalse(should_mock_download("config.json", file_size=1024))  # 1KB
        self.assertFalse(should_mock_download("vocab.txt", file_size=1024 * 1024))  # 1MB
        
        # Large file (>= 1GB) - should mock
        self.assertTrue(should_mock_download("model.bin", file_size=MOCK_SIZE_THRESHOLD))  # 1GB
        self.assertTrue(should_mock_download("model.bin", file_size=MOCK_SIZE_THRESHOLD + 1))  # > 1GB

    def test_should_mock_download_unknown_files(self):
        """Test that unknown file types default to mock for safety"""
        # Unknown file types should be mocked for safety
        self.assertTrue(should_mock_download("unknown_file.xyz"))
        self.assertTrue(should_mock_download("data.dat"))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""

    def test_get_downloader_default(self):
        """Test get_downloader with default source"""
        downloader = get_downloader()
        self.assertIsInstance(downloader, ModelDownloader)
        self.assertEqual(downloader.source, "baai_modelhub")

    def test_get_downloader_custom_source(self):
        """Test get_downloader with custom source"""
        downloader = get_downloader(source="huggingface")
        self.assertIsInstance(downloader, ModelDownloader)
        self.assertEqual(downloader.source, "huggingface")

    def test_get_downloader_with_mirror(self):
        """Test get_downloader with HuggingFace mirror"""
        with patch.dict(os.environ, {}, clear=True):
            downloader = get_downloader(
                source="huggingface_mirror",
                hf_mirror="https://hf-mirror.com"
            )
            self.assertIsInstance(downloader, ModelDownloader)
            self.assertEqual(downloader.hf_endpoint, "https://hf-mirror.com")

    @patch('flagai.model.download_sources.get_downloader')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_test_download_source_quick(self, mock_getsize, mock_exists, mock_get_downloader):
        """Test test_download_source_quick function - Mocked"""
        if test_download_source_quick is None:
            self.skipTest("test_download_source_quick not available")
        
        mock_downloader = MagicMock()
        mock_file_path = "/path/to/file"
        mock_downloader.download_file.return_value = mock_file_path
        mock_get_downloader.return_value = mock_downloader
        
        # Mock file existence and size checks
        mock_exists.return_value = True
        mock_getsize.return_value = 100  # File size > 0
        
        result = test_download_source_quick(
            source="huggingface",
            model_name="test-model",
            file_name="config.json"
        )
        self.assertTrue(result)
        mock_downloader.download_file.assert_called_once()

    @patch('flagai.model.download_sources.ModelDownloader')
    def test_test_download_source_quick_failure(self, mock_downloader_class):
        """Test test_download_source_quick with download failure - Mocked"""
        if test_download_source_quick is None:
            self.skipTest("test_download_source_quick not available")
        
        mock_downloader = MagicMock()
        mock_downloader.download_file.side_effect = Exception("Download failed")
        mock_downloader_class.return_value = mock_downloader
        
        result = test_download_source_quick(
            source="huggingface",
            model_name="test-model",
            file_name="config.json"
        )
        self.assertFalse(result)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestModelDownloader))
    suite.addTest(unittest.makeSuite(TestDownloadStrategy))
    suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

