#!/usr/bin/env python3
"""Quick diagnostic script to check GPU availability and Ollama configuration."""

import subprocess
import os
import sys

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def check_ollama_env():
    """Check Ollama environment variables."""
    print("\nOllama Environment Variables:")
    gpu_vars = ['OLLAMA_GPU_LAYERS', 'OLLAMA_NUM_GPU', 'CUDA_VISIBLE_DEVICES']
    found = False
    for var in gpu_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var} = {value}")
            found = True
    if not found:
        print("  No GPU-related environment variables set")
    return found

def check_ollama_info():
    """Check Ollama server info."""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/version', timeout=2)
        if response.status_code == 200:
            print("\n✓ Ollama server is running")
            print(f"  Version info: {response.json()}")
            return True
        else:
            print(f"\n✗ Ollama server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"\n✗ Cannot connect to Ollama server: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False

def main():
    print("=" * 60)
    print("Ollama GPU Diagnostic Check")
    print("=" * 60)
    
    has_gpu = check_nvidia_gpu()
    has_env = check_ollama_env()
    ollama_running = check_ollama_info()
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    if not has_gpu:
        print("1. Install NVIDIA GPU drivers if you have an NVIDIA GPU")
        print("2. For AMD/Intel GPUs, check Ollama documentation for ROCm/oneAPI support")
    else:
        if not has_env:
            print("1. Set OLLAMA_GPU_LAYERS environment variable:")
            print("   Windows PowerShell: $env:OLLAMA_GPU_LAYERS='35'")
            print("   Windows CMD: set OLLAMA_GPU_LAYERS=35")
            print("   Linux/Mac: export OLLAMA_GPU_LAYERS=35")
            print("\n2. Restart Ollama after setting environment variables")
            print("3. For phi3:3.8b, try OLLAMA_GPU_LAYERS=35 (or higher if you have more VRAM)")
    
    if ollama_running:
        print("\n4. Check current GPU usage: ollama ps")
        print("5. Try pulling the model again: ollama pull phi3:3.8b")
        print("6. Monitor GPU usage during inference: nvidia-smi -l 1")

if __name__ == "__main__":
    main()
