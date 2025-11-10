#!/usr/bin/env python3
"""Quick diagnostic script to check GPU availability and Ollama configuration for macOS."""

import subprocess
import os
import sys
import platform

def check_apple_silicon():
    """Check if running on Apple Silicon (M1/M2/M3)."""
    try:
        # Check architecture
        arch = platform.machine()
        if arch == 'arm64':
            # Get CPU brand to confirm Apple Silicon
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                cpu_brand = result.stdout.strip()
                print(f"✓ Apple Silicon detected: {cpu_brand}")
                print(f"  Architecture: {arch}")
                return True
        else:
            print(f"  Architecture: {arch} (Intel Mac)")
        return False
    except Exception as e:
        print(f"✗ Error checking CPU: {e}")
        return False

def check_gpu_info():
    """Check GPU information using system_profiler."""
    try:
        result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("\n✓ GPU Information:")
            # Extract key GPU info
            lines = result.stdout.split('\n')
            gpu_lines = [line for line in lines if 'Chipset Model' in line or 'Metal' in line or 'VRAM' in line]
            if gpu_lines:
                for line in gpu_lines[:5]:  # Show first 5 relevant lines
                    print(f"  {line.strip()}")
            else:
                print("  " + result.stdout[:200])  # Show first 200 chars if no specific matches
            return True
        else:
            print("✗ Could not retrieve GPU information")
            return False
    except FileNotFoundError:
        print("✗ system_profiler not found")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def check_ollama_env():
    """Check Ollama environment variables for macOS."""
    print("\nOllama Environment Variables:")
    gpu_vars = ['OLLAMA_NUM_GPU', 'OLLAMA_GPU_LAYERS', 'OLLAMA_FLASH_ATTENTION']
    found = False
    for var in gpu_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var} = {value}")
            found = True
    if not found:
        print("  No GPU-related environment variables set")
        print("  Note: On Apple Silicon, Ollama uses Metal automatically (no config needed)")
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
    print("Ollama GPU Diagnostic Check (macOS)")
    print("=" * 60)
    
    is_apple_silicon = check_apple_silicon()
    has_gpu_info = check_gpu_info()
    has_env = check_ollama_env()
    ollama_running = check_ollama_info()
    
    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    if is_apple_silicon:
        print("✓ Apple Silicon detected - Ollama will use Metal GPU acceleration automatically")
        print("\n1. No additional configuration needed for GPU acceleration")
        print("2. Ollama uses Metal framework for GPU acceleration on Apple Silicon")
        print("3. For better performance, ensure you have enough unified memory (RAM)")
    else:
        print("ℹ Intel Mac detected")
        print("\n1. Ollama will use CPU or Metal (if supported by your GPU)")
        print("2. Check if your Mac supports Metal: https://support.apple.com/en-us/HT202823")
    
    if not has_env and is_apple_silicon:
        print("\n4. Optional: Set OLLAMA_NUM_GPU to control GPU usage:")
        print("   export OLLAMA_NUM_GPU=1")
        print("   (Usually not needed - Ollama detects GPU automatically)")
    
    if ollama_running:
        print("\n5. Check current model usage: ollama ps")
        print("6. Try pulling the model: ollama pull phi3:3.8b")
        print("7. Monitor system performance: Activity Monitor → Window → GPU History")
        print("8. Test inference speed: ollama run phi3:3.8b")

if __name__ == "__main__":
    main()
