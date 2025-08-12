#!/usr/bin/env python3
"""
Debug script to isolate the issue with the SPARC analysis
"""

print("=== DEBUGGING SPARC SCRIPT ===")

# Test 1: Basic imports
print("Testing basic imports...")
try:
    import numpy as np
    print("✓ numpy imported")
except Exception as e:
    print(f"✗ numpy failed: {e}")
    exit(1)

try:
    import pandas as pd
    print("✓ pandas imported")
except Exception as e:
    print(f"✗ pandas failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib imported")
except Exception as e:
    print(f"✗ matplotlib failed: {e}")

# Test 2: GPU imports
print("\nTesting GPU imports...")
try:
    import cupy as cp
    print("✓ CuPy imported successfully")
    
    # Test GPU access
    try:
        device = cp.cuda.Device()
        print(f"✓ GPU device accessible: {device}")
        mem_info = device.mem_info
        print(f"✓ GPU memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
        
        # Test simple GPU operation
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        print(f"✓ GPU computation test: sum([1,2,3,4,5]) = {result}")
        
    except Exception as e:
        print(f"✗ GPU access failed: {e}")
        
except ImportError:
    print("✗ CuPy not available")

# Test 3: File system access
print("\nTesting file system...")
sparc_directory = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"

try:
    import os
    import glob
    
    print(f"✓ Directory exists: {os.path.exists(sparc_directory)}")
    print(f"✓ Directory path: {sparc_directory}")
    
    # Look for .dat files
    dat_files = glob.glob(os.path.join(sparc_directory, "*.dat"))
    print(f"✓ Found {len(dat_files)} .dat files")
    
    if len(dat_files) > 0:
        print(f"✓ First few files: {dat_files[:3]}")
        
        # Try reading one file
        try:
            with open(dat_files[0], 'r') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            print(f"✓ Sample file content:")
            for i, line in enumerate(first_lines):
                print(f"   Line {i+1}: {line[:50]}{'...' if len(line) > 50 else ''}")
        except Exception as e:
            print(f"✗ File reading failed: {e}")
    
except Exception as e:
    print(f"✗ File system test failed: {e}")

# Test 4: scipy imports
print("\nTesting scipy imports...")
try:
    from scipy.optimize import minimize
    print("✓ scipy.optimize imported")
except Exception as e:
    print(f"✗ scipy.optimize failed: {e}")

try:
    from scipy.stats import pearsonr
    print("✓ scipy.stats imported")
except Exception as e:
    print(f"✗ scipy.stats failed: {e}")

# Test 5: Simple function definition
print("\nTesting function definitions...")
try:
    def simple_geodesic_test(r):
        return np.sqrt(1.0 / r)
    
    test_r = np.array([1.0, 2.0, 3.0])
    test_result = simple_geodesic_test(test_r)
    print(f"✓ Simple function test: {test_result}")
    
except Exception as e:
    print(f"✗ Function test failed: {e}")

print("\n=== DEBUG COMPLETE ===")
print("If all tests pass, the issue is likely in the main script logic.")
print("If any test fails, that's where the problem is.")