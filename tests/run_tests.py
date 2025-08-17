"""
Test runner for the ML pipeline project.
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def run_all_tests():
    """Run all tests in the test suite."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    # Return success/failure
    return len(result.failures) == 0 and len(result.errors) == 0


def run_specific_test(test_module):
    """
    Run tests from a specific module.
    
    Args:
        test_module (str): Name of the test module (e.g., 'test_data_ingestion')
    """
    try:
        # Import the specific test module
        module = __import__(test_module)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except ImportError as e:
        print(f"Error importing test module '{test_module}': {e}")
        return False


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for ML Pipeline project')
    parser.add_argument(
        '--module', '-m',
        help='Run tests from specific module (e.g., test_data_ingestion)',
        default=None
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Run tests with coverage report'
    )
    
    args = parser.parse_args()
    
    if args.coverage:
        try:
            import coverage
            
            # Start coverage
            cov = coverage.Coverage(source=['src'])
            cov.start()
            
            # Run tests
            if args.module:
                success = run_specific_test(args.module)
            else:
                success = run_all_tests()
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            print("\n" + "="*70)
            print("COVERAGE REPORT")
            print("="*70)
            cov.report(show_missing=True)
            
            # Generate HTML coverage report
            html_dir = os.path.join(project_root, 'htmlcov')
            cov.html_report(directory=html_dir)
            print(f"\nDetailed HTML coverage report generated in: {html_dir}")
            
        except ImportError:
            print("Coverage package not installed. Install with: pip install coverage")
            print("Running tests without coverage...")
            
            if args.module:
                success = run_specific_test(args.module)
            else:
                success = run_all_tests()
    else:
        if args.module:
            success = run_specific_test(args.module)
        else:
            success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()