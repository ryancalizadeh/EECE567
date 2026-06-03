"""Quick integration test for admm_test with OPF-based ICs"""
import sys
sys.path.insert(0, '.')

# Run a short ADMM test with 12 buses
print("Testing admm_test with OPF-based initial conditions...")
print("=" * 60)

try:
    from saap import admm_test
    # Run with just 1 test case (no parallel comparison) for speed
    admm_test(n_buses=4, seq_and_parallel=False)
    print("\n" + "=" * 60)
    print("Integration test PASSED!")
except Exception as e:
    print(f"\nIntegration test FAILED with error:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
