"""
Test Download Sources Module
Provides testing functions for different model download sources

This module provides utilities to test the availability and functionality
of different download sources (BAAI ModelHub, HuggingFace, HuggingFace mirrors, ModelScope).
"""
import os
import tempfile
import shutil
from typing import List, Dict, Optional, Tuple
from flagai.logger import log_dist


def test_download_source(source: str,
                        model_name: str,
                        test_files: Optional[List[str]] = None,
                        hf_mirror: Optional[str] = None,
                        use_auth_token: Optional[str] = None,
                        cleanup: bool = True) -> Dict[str, any]:
    """
    Test a specific download source.
    
    Args:
        source: Download source to test ("baai_modelhub", "huggingface", 
                "huggingface_mirror", "modelscope")
        model_name: Name of the model to test
        test_files: List of files to test download (defaults to common files)
        hf_mirror: HuggingFace mirror URL (for huggingface_mirror source)
        use_auth_token: HuggingFace authentication token
        cleanup: Whether to cleanup downloaded files after test
    
    Returns:
        Dictionary with test results including:
        - source: Source name
        - model_name: Model name
        - success: Whether test was successful
        - files_tested: List of files tested
        - files_success: List of successfully downloaded files
        - files_failed: List of failed files
        - errors: List of error messages
        - download_time: Time taken for downloads
    """
    import time
    from flagai.model.download_sources import get_downloader
    
    results = {
        "source": source,
        "model_name": model_name,
        "success": False,
        "files_tested": [],
        "files_success": [],
        "files_failed": [],
        "errors": [],
        "download_time": 0.0
    }
    
    # Default test files if not provided
    if test_files is None:
        test_files = ["config.json"]
        # Add more common files based on source
        if source in ["huggingface", "huggingface_mirror"]:
            test_files.extend(["tokenizer.json", "tokenizer_config.json"])
        elif source == "modelscope":
            test_files.extend(["configuration.json"])
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix=f"flagai_test_{source}_")
    
    try:
        # Get downloader
        downloader = get_downloader(
            source=source,
            hf_mirror=hf_mirror,
            use_auth_token=use_auth_token
        )
        
        # Test listing files first
        try:
            files_list = downloader.list_files(model_name)
            if files_list:
                results["files_available"] = len(files_list)
                print(f"✓ {source}: Found {len(files_list)} files for {model_name}")
            else:
                print(f"⚠ {source}: No files found for {model_name}")
        except Exception as e:
            results["errors"].append(f"Failed to list files: {str(e)}")
            print(f"✗ {source}: Failed to list files - {str(e)}")
        
        # Test downloading files
        start_time = time.time()
        for file_name in test_files:
            results["files_tested"].append(file_name)
            try:
                file_path = downloader.download_file(
                    model_name=model_name,
                    file_name=file_name,
                    download_path=temp_dir,
                    rank=0
                )
                
                # Check if file exists and has content
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 0:
                        results["files_success"].append(file_name)
                        print(f"  ✓ Downloaded {file_name} ({file_size} bytes)")
                    else:
                        results["files_failed"].append(file_name)
                        results["errors"].append(f"{file_name}: File is empty")
                        print(f"  ✗ {file_name}: File is empty")
                else:
                    results["files_failed"].append(file_name)
                    results["errors"].append(f"{file_name}: File not found after download")
                    print(f"  ✗ {file_name}: File not found after download")
                    
            except Exception as e:
                results["files_failed"].append(file_name)
                error_msg = f"{file_name}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"  ✗ {file_name}: {str(e)}")
        
        results["download_time"] = time.time() - start_time
        
        # Determine overall success
        if len(results["files_success"]) > 0:
            results["success"] = True
            print(f"✓ {source}: Test completed - {len(results['files_success'])}/{len(test_files)} files downloaded successfully")
        else:
            results["success"] = False
            print(f"✗ {source}: Test failed - No files downloaded successfully")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Test setup failed: {str(e)}")
        print(f"✗ {source}: Test setup failed - {str(e)}")
    
    finally:
        # Cleanup
        if cleanup and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"  Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"  Warning: Failed to cleanup {temp_dir}: {str(e)}")
    
    return results


def test_all_sources(model_name: str,
                    test_files: Optional[List[str]] = None,
                    hf_mirror: Optional[str] = None,
                    use_auth_token: Optional[str] = None,
                    sources: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Test all available download sources.
    
    Args:
        model_name: Name of the model to test
        test_files: List of files to test download
        hf_mirror: HuggingFace mirror URL
        use_auth_token: HuggingFace authentication token
        sources: List of sources to test (defaults to all available)
    
    Returns:
        Dictionary mapping source names to test results
    """
    if sources is None:
        sources = ["baai_modelhub", "huggingface", "huggingface_mirror", "modelscope"]
    
    print(f"\n{'='*60}")
    print(f"Testing Download Sources for Model: {model_name}")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for source in sources:
        print(f"\n--- Testing {source.upper()} ---")
        try:
            result = test_download_source(
                source=source,
                model_name=model_name,
                test_files=test_files,
                hf_mirror=hf_mirror,
                use_auth_token=use_auth_token,
                cleanup=True
            )
            all_results[source] = result
        except Exception as e:
            print(f"✗ {source}: Failed to run test - {str(e)}")
            all_results[source] = {
                "source": source,
                "success": False,
                "errors": [f"Test execution failed: {str(e)}"]
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}\n")
    
    for source, result in all_results.items():
        status = "✓ PASS" if result.get("success", False) else "✗ FAIL"
        files_success = len(result.get("files_success", []))
        files_tested = len(result.get("files_tested", []))
        print(f"{status} {source:20s} - {files_success}/{files_tested} files downloaded")
        if result.get("download_time"):
            print(f"      Download time: {result['download_time']:.2f}s")
        if result.get("errors"):
            for error in result["errors"][:3]:  # Show first 3 errors
                print(f"      Error: {error}")
    
    return all_results


def test_source_availability(sources: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Test if download sources are available (can connect).
    
    Args:
        sources: List of sources to test (defaults to all available)
    
    Returns:
        Dictionary mapping source names to availability status
    """
    if sources is None:
        sources = ["baai_modelhub", "huggingface", "huggingface_mirror", "modelscope"]
    
    print(f"\n{'='*60}")
    print("Testing Download Source Availability")
    print(f"{'='*60}\n")
    
    availability = {}
    
    for source in sources:
        try:
            from flagai.model.download_sources import get_downloader
            
            downloader = get_downloader(source=source)
            
            # Try to create downloader (this tests imports and basic setup)
            if downloader:
                availability[source] = True
                print(f"✓ {source}: Available")
            else:
                availability[source] = False
                print(f"✗ {source}: Not available")
                
        except ImportError as e:
            availability[source] = False
            print(f"✗ {source}: Not available - Missing dependency: {str(e)}")
        except Exception as e:
            availability[source] = False
            print(f"✗ {source}: Not available - {str(e)}")
    
    return availability


def compare_download_sources(model_name: str,
                            test_files: Optional[List[str]] = None,
                            hf_mirror: Optional[str] = None,
                            use_auth_token: Optional[str] = None) -> None:
    """
    Compare performance and availability of different download sources.
    
    Args:
        model_name: Name of the model to test
        test_files: List of files to test download
        hf_mirror: HuggingFace mirror URL
        use_auth_token: HuggingFace authentication token
    """
    print(f"\n{'='*60}")
    print(f"Comparing Download Sources for: {model_name}")
    print(f"{'='*60}\n")
    
    # Test availability first
    availability = test_source_availability()
    
    # Filter available sources
    available_sources = [s for s, avail in availability.items() if avail]
    
    if not available_sources:
        print("\n✗ No download sources are available!")
        return
    
    print(f"\nTesting {len(available_sources)} available sources...\n")
    
    # Test all available sources
    results = test_all_sources(
        model_name=model_name,
        test_files=test_files,
        hf_mirror=hf_mirror,
        use_auth_token=use_auth_token,
        sources=available_sources
    )
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("Performance Comparison")
    print(f"{'='*60}\n")
    print(f"{'Source':<20} {'Status':<10} {'Files':<10} {'Time (s)':<12} {'Speed'}")
    print("-" * 60)
    
    for source in available_sources:
        result = results.get(source, {})
        status = "✓ PASS" if result.get("success", False) else "✗ FAIL"
        files_success = len(result.get("files_success", []))
        files_tested = len(result.get("files_tested", []))
        files_str = f"{files_success}/{files_tested}"
        time_taken = result.get("download_time", 0.0)
        
        # Calculate speed (files per second)
        if time_taken > 0:
            speed = f"{files_success/time_taken:.2f} files/s"
        else:
            speed = "N/A"
        
        print(f"{source:<20} {status:<10} {files_str:<10} {time_taken:<12.2f} {speed}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test download sources")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model name to test")
    parser.add_argument("--source", type=str, default=None,
                       choices=["baai_modelhub", "huggingface", "huggingface_mirror", "modelscope"],
                       help="Specific source to test (default: all)")
    parser.add_argument("--files", type=str, nargs="+", default=None,
                       help="Files to test download")
    parser.add_argument("--hf-mirror", type=str, default=None,
                       help="HuggingFace mirror URL")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all available sources")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_download_sources(
            model_name=args.model,
            test_files=args.files,
            hf_mirror=args.hf_mirror
        )
    elif args.source:
        result = test_download_source(
            source=args.source,
            model_name=args.model,
            test_files=args.files,
            hf_mirror=args.hf_mirror
        )
        print(f"\nTest Result: {'PASS' if result['success'] else 'FAIL'}")
    else:
        test_all_sources(
            model_name=args.model,
            test_files=args.files,
            hf_mirror=args.hf_mirror
        )

