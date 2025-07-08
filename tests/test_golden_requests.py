"""Golden request unit tests for Arc Runtime.

These tests verify that Arc correctly intercepts and modifies problematic requests
based on canonical examples defined in golden_requests.yaml.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import yaml

# Add parent directory to path to import runtime
sys.path.insert(0, str(Path(__file__).parent.parent))
from runtime.arc import Arc
from runtime.interceptors.base import BaseInterceptor
from runtime.patterns.registry import PatternRegistry


class TestGoldenRequestsPatternMatching(unittest.TestCase):
    """Test Arc Runtime pattern matching against golden request examples."""

    @classmethod
    def setUpClass(cls):
        """Load golden request test cases from YAML."""
        fixtures_path = Path(__file__).parent / "fixtures" / "golden_requests.yaml"
        with open(fixtures_path, "r") as f:
            cls.golden_data = yaml.safe_load(f)
        cls.test_cases = cls.golden_data["test_cases"]

    def setUp(self):
        """Set up test fixtures."""
        # Create pattern registry directly
        self.registry = PatternRegistry()

    def test_golden_request_patterns(self):
        """Test pattern matching for each golden request case."""
        for test_case in self.test_cases:
            with self.subTest(name=test_case["name"]):
                request = test_case["request"]
                expected_fix = test_case["expected_fix"]

                # Test pattern matching
                actual_fix = self.registry.match(request)

                if expected_fix is None:
                    self.assertIsNone(
                        actual_fix, f"No fix should be applied for {test_case['name']}"
                    )
                else:
                    self.assertIsNotNone(
                        actual_fix, f"Fix should be applied for {test_case['name']}"
                    )
                    self.assertEqual(
                        actual_fix,
                        expected_fix,
                        f"Wrong fix applied for {test_case['name']}",
                    )

    def test_interception_overhead(self):
        """Test that Arc's pattern matching overhead is less than 5ms."""
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.95,
        }

        # Warm up
        for _ in range(100):
            self.registry.match(request)

        # Measure pattern matching time
        num_iterations = 1000
        start_time = time.time()
        for _ in range(num_iterations):
            fix = self.registry.match(request)
        match_time = time.time() - start_time

        # Calculate average overhead per request
        overhead_ms = (match_time / num_iterations) * 1000

        # Assert overhead is less than 5ms
        self.assertLess(
            overhead_ms,
            5.0,
            f"Pattern matching overhead ({overhead_ms:.2f}ms) exceeds 5ms requirement",
        )


class TestGoldenRequestsWithRealClient(unittest.TestCase):
    """Test Arc Runtime with real OpenAI client (requires OPENAI_API_KEY).

    Note: These tests are skipped if OPENAI_API_KEY is not set.
    To run these tests:
        export OPENAI_API_KEY=your_key
        python -m unittest tests.test_golden_requests.TestGoldenRequestsWithRealClient -v
    """

    @classmethod
    def setUpClass(cls):
        """Load golden request test cases from YAML."""
        fixtures_path = Path(__file__).parent / "fixtures" / "golden_requests.yaml"
        with open(fixtures_path, "r") as f:
            cls.golden_data = yaml.safe_load(f)
        cls.test_cases = cls.golden_data["test_cases"]

    def setUp(self):
        """Set up test fixtures."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY not set - skipping integration tests")

        # Create Arc instance
        self.arc = Arc()

    @unittest.skip("Requires real OpenAI client - see class docstring for how to run")
    def test_golden_requests_with_real_client(self):
        """Test each golden request case with real OpenAI client.

        This test is skipped by default. To run:
        1. Set OPENAI_API_KEY environment variable
        2. Remove the @unittest.skip decorator
        3. Note: This will make real API calls!
        """
        pass  # Implementation would go here


class TestGoldenRequestsIntegration(unittest.TestCase):
    """Integration tests for golden requests."""

    def test_yaml_bundle_loadable(self):
        """Verify the YAML bundle can be loaded and is valid."""
        fixtures_path = Path(__file__).parent / "fixtures" / "golden_requests.yaml"
        self.assertTrue(fixtures_path.exists(), "golden_requests.yaml should exist")

        # Load and validate structure
        with open(fixtures_path, "r") as f:
            data = yaml.safe_load(f)

        self.assertIn("test_cases", data)
        self.assertIsInstance(data["test_cases"], list)
        self.assertGreaterEqual(
            len(data["test_cases"]), 3, "Should have at least 3 test cases"
        )

        # Validate each test case structure
        for test_case in data["test_cases"]:
            self.assertIn("name", test_case)
            self.assertIn("description", test_case)
            self.assertIn("request", test_case)
            self.assertIn("expected_fix", test_case)

            # Validate request structure
            request = test_case["request"]
            self.assertIn("model", request)
            self.assertIn("messages", request)


if __name__ == "__main__":
    unittest.main()
