# tests/test_runner.py
#!/usr/bin/env python3
"""
Test runner for Soma ML Platform
"""

import subprocess
import sys
from pathlib import Path


def run_test_suite(suite_name, test_path, markers=None):
    """Run a specific test suite."""
    print(f"\n🧪 Running {suite_name} Tests...")

    cmd = ["pytest", str(test_path), "-v"]

    if markers:
        cmd.extend(["-m", markers])

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {suite_name} tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {suite_name} tests failed!")
        return False


def main():
    """Main test runner."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"

    print("🚀 Soma ML Platform Test Suite")
    print("=" * 50)

    # Set up environment
    import os

    os.environ["PYTHONPATH"] = str(project_root)
    os.environ["ENVIRONMENT"] = "testing"

    results = {}

    # Run test suites
    test_suites = [
        ("Unit", tests_dir / "unit", "unit"),
        ("Integration", tests_dir / "integration", "integration"),
        ("E2E", tests_dir / "e2e", "e2e"),
    ]

    for name, path, marker in test_suites:
        if path.exists():
            results[name] = run_test_suite(name, path, marker)
        else:
            print(f"⚠️  {name} test directory not found: {path}")
            results[name] = None

    # Generate coverage report
    print("\n📊 Generating Coverage Report...")
    try:
        subprocess.run(
            [
                "pytest",
                str(tests_dir),
                "--cov=src",
                "--cov-report=html:tests/coverage_html",
                "--cov-report=term-missing",
            ],
            check=True,
        )
        print("✅ Coverage report generated!")
    except subprocess.CalledProcessError:
        print("❌ Coverage report generation failed!")

    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)

    total_suites = len([r for r in results.values() if r is not None])
    passed_suites = len([r for r in results.values() if r is True])

    for suite, result in results.items():
        if result is True:
            print(f"✅ {suite}: PASSED")
        elif result is False:
            print(f"❌ {suite}: FAILED")
        else:
            print(f"⚠️  {suite}: SKIPPED")

    print(f"\nOverall: {passed_suites}/{total_suites} suites passed")

    # Exit with appropriate code
    if passed_suites == total_suites and total_suites > 0:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
