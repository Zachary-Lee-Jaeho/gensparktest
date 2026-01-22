"""
GitHub-based LLVM Source Fetcher.

This module provides a fast way to fetch specific LLVM files directly
from GitHub without cloning the entire repository.

Features:
- Direct file download via GitHub raw URLs
- Caching of downloaded files
- Parallel downloads
- No git required
"""

import os
import json
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


# GitHub raw content base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/llvm/llvm-project"

# Key files for each backend module
BACKEND_KEY_FILES = {
    "MCCodeEmitter": [
        "MCTargetDesc/{backend}MCCodeEmitter.cpp",
        "MCTargetDesc/{backend}MCCodeEmitter.h",
    ],
    "ELFObjectWriter": [
        "MCTargetDesc/{backend}ELFObjectWriter.cpp",
        "MCTargetDesc/{backend}ELFObjectWriter.h",
    ],
    "AsmPrinter": [
        "{backend}AsmPrinter.cpp",
        "{backend}AsmPrinter.h",
        "MCTargetDesc/{backend}InstPrinter.cpp",
    ],
    "ISelDAGToDAG": [
        "{backend}ISelDAGToDAG.cpp",
        "{backend}ISelDAGToDAG.h",
    ],
    "ISelLowering": [
        "{backend}ISelLowering.cpp",
        "{backend}ISelLowering.h",
    ],
    "RegisterInfo": [
        "{backend}RegisterInfo.cpp",
        "{backend}RegisterInfo.h",
    ],
    "InstrInfo": [
        "{backend}InstrInfo.cpp",
        "{backend}InstrInfo.h",
    ],
    "Subtarget": [
        "{backend}Subtarget.cpp",
        "{backend}Subtarget.h",
    ],
}


@dataclass
class FetchedFile:
    """Represents a fetched file."""
    path: str           # Relative path in LLVM
    content: str        # File content
    size: int           # Size in bytes
    module: str         # Module name
    backend: str        # Backend name
    cached: bool        # Whether from cache


class GitHubLLVMFetcher:
    """
    Fetches LLVM files directly from GitHub.
    
    Much faster than git clone for extracting specific files.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        version: str = "llvmorg-18.1.0",
        verbose: bool = True
    ):
        """
        Initialize the fetcher.
        
        Args:
            cache_dir: Directory to cache downloaded files
            version: LLVM version tag
            verbose: Print progress
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vega-verified" / "llvm-files"
        self.version = version
        self.verbose = verbose
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            "files_downloaded": 0,
            "files_cached": 0,
            "bytes_downloaded": 0,
            "errors": 0,
        }
    
    def _get_raw_url(self, file_path: str) -> str:
        """Get GitHub raw URL for a file."""
        return f"{GITHUB_RAW_BASE}/{self.version}/llvm/lib/Target/{file_path}"
    
    def _get_cache_path(self, file_path: str) -> Path:
        """Get local cache path for a file."""
        # Create hash of path for flat storage
        safe_name = file_path.replace('/', '_')
        return self.cache_dir / self.version / safe_name
    
    def _download_file(self, url: str) -> Optional[str]:
        """Download a file from URL."""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'VEGA-Verified/1.0')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8', errors='ignore')
                self.stats["bytes_downloaded"] += len(content)
                return content
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # File not found is expected for some paths
            if self.verbose:
                print(f"HTTP Error {e.code} for {url}")
            self.stats["errors"] += 1
            return None
        except Exception as e:
            if self.verbose:
                print(f"Download error for {url}: {e}")
            self.stats["errors"] += 1
            return None
    
    def fetch_file(self, backend: str, relative_path: str, module: str = "Unknown") -> Optional[FetchedFile]:
        """
        Fetch a single file.
        
        Args:
            backend: Backend name (e.g., "RISCV")
            relative_path: Path relative to backend directory
            module: Module name for metadata
            
        Returns:
            FetchedFile or None if not found
        """
        full_path = f"{backend}/{relative_path}"
        cache_path = self._get_cache_path(full_path)
        
        # Check cache
        if cache_path.exists():
            content = cache_path.read_text(encoding='utf-8', errors='ignore')
            self.stats["files_cached"] += 1
            return FetchedFile(
                path=full_path,
                content=content,
                size=len(content),
                module=module,
                backend=backend,
                cached=True,
            )
        
        # Download
        url = self._get_raw_url(full_path)
        content = self._download_file(url)
        
        if content is None:
            return None
        
        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding='utf-8')
        
        self.stats["files_downloaded"] += 1
        
        return FetchedFile(
            path=full_path,
            content=content,
            size=len(content),
            module=module,
            backend=backend,
            cached=False,
        )
    
    def fetch_backend_files(
        self,
        backend: str,
        modules: Optional[List[str]] = None,
        parallel: bool = True,
        max_workers: int = 8
    ) -> Dict[str, List[FetchedFile]]:
        """
        Fetch all key files for a backend.
        
        Args:
            backend: Backend name (e.g., "RISCV")
            modules: List of modules to fetch (None = all)
            parallel: Use parallel downloads
            max_workers: Number of parallel workers
            
        Returns:
            Dict mapping module names to list of fetched files
        """
        if modules is None:
            modules = list(BACKEND_KEY_FILES.keys())
        
        if self.verbose:
            print(f"Fetching {backend} backend files for modules: {modules}")
        
        # Build list of files to fetch
        files_to_fetch = []
        for module in modules:
            patterns = BACKEND_KEY_FILES.get(module, [])
            for pattern in patterns:
                path = pattern.format(backend=backend)
                files_to_fetch.append((backend, path, module))
        
        results: Dict[str, List[FetchedFile]] = {m: [] for m in modules}
        
        if parallel and len(files_to_fetch) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.fetch_file, b, p, m): (b, p, m)
                    for b, p, m in files_to_fetch
                }
                
                for future in as_completed(futures):
                    b, p, m = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results[m].append(result)
                            if self.verbose:
                                status = "cached" if result.cached else "downloaded"
                                print(f"  [{status}] {p}")
                    except Exception as e:
                        if self.verbose:
                            print(f"  [error] {p}: {e}")
        else:
            for b, p, m in files_to_fetch:
                result = self.fetch_file(b, p, m)
                if result:
                    results[m].append(result)
                    if self.verbose:
                        status = "cached" if result.cached else "downloaded"
                        print(f"  [{status}] {p}")
        
        return results
    
    def fetch_additional_files(
        self,
        backend: str,
        patterns: List[str]
    ) -> List[FetchedFile]:
        """
        Fetch additional files matching patterns.
        
        Note: This requires knowing exact file paths since we can't
        list directories via raw GitHub URLs.
        """
        results = []
        for pattern in patterns:
            path = pattern.format(backend=backend)
            result = self.fetch_file(backend, path, "Other")
            if result:
                results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """Get fetch statistics."""
        return dict(self.stats)
    
    def clear_cache(self, backend: Optional[str] = None) -> None:
        """Clear cached files."""
        import shutil
        
        if backend:
            # Clear specific backend
            for f in self.cache_dir.glob(f"**/{backend}_*"):
                f.unlink()
        else:
            # Clear all
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# Alternative: Use GitHub API to list directory contents
class GitHubAPIFetcher:
    """
    Uses GitHub API to list and fetch files.
    
    Slower due to API rate limits but can discover files.
    """
    
    API_BASE = "https://api.github.com/repos/llvm/llvm-project/contents"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        version: str = "llvmorg-18.1.0",
        verbose: bool = True,
        token: Optional[str] = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "vega-verified" / "llvm-api"
        self.version = version
        self.verbose = verbose
        self.token = token or os.environ.get("GITHUB_TOKEN")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _api_request(self, path: str) -> Optional[Dict]:
        """Make GitHub API request."""
        url = f"{self.API_BASE}/{path}?ref={self.version}"
        
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/vnd.github.v3+json')
            req.add_header('User-Agent', 'VEGA-Verified/1.0')
            
            if self.token:
                req.add_header('Authorization', f'token {self.token}')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            if self.verbose:
                print(f"API error for {path}: {e}")
            return None
    
    def list_backend_files(self, backend: str) -> List[str]:
        """List all files in a backend directory."""
        files = []
        
        base_path = f"llvm/lib/Target/{backend}"
        result = self._api_request(base_path)
        
        if not result:
            return files
        
        # Process directory listing
        for item in result:
            if item["type"] == "file" and item["name"].endswith(('.cpp', '.h')):
                files.append(item["path"])
            elif item["type"] == "dir":
                # Recursively list subdirectories
                subdir_files = self._list_directory(item["path"])
                files.extend(subdir_files)
        
        return files
    
    def _list_directory(self, path: str) -> List[str]:
        """Recursively list directory contents."""
        files = []
        result = self._api_request(path)
        
        if not result:
            return files
        
        for item in result:
            if item["type"] == "file" and item["name"].endswith(('.cpp', '.h')):
                files.append(item["path"])
            elif item["type"] == "dir":
                files.extend(self._list_directory(item["path"]))
        
        return files
    
    def fetch_file_content(self, file_path: str) -> Optional[str]:
        """Fetch content of a specific file."""
        # Use raw URL for content (API has size limits)
        raw_url = f"https://raw.githubusercontent.com/llvm/llvm-project/{self.version}/{file_path}"
        
        try:
            req = urllib.request.Request(raw_url)
            req.add_header('User-Agent', 'VEGA-Verified/1.0')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            if self.verbose:
                print(f"Error fetching {file_path}: {e}")
            return None


# Quick test
if __name__ == "__main__":
    fetcher = GitHubLLVMFetcher(verbose=True)
    
    # Test fetching RISC-V backend
    print("\nFetching RISC-V backend files...")
    files = fetcher.fetch_backend_files("RISCV", modules=["MCCodeEmitter", "ELFObjectWriter"])
    
    print(f"\n=== Results ===")
    for module, module_files in files.items():
        print(f"\n{module}: {len(module_files)} files")
        for f in module_files:
            print(f"  - {f.path} ({f.size} bytes)")
    
    print(f"\nStats: {fetcher.get_stats()}")
