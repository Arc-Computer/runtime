"""
Comprehensive performance test for Arc Runtime
Verifies <5ms interception overhead (P99) requirement from PRD
"""

import concurrent.futures
import json
import statistics
import threading
import time
from typing import Any, Callable, Dict, List
from unittest.mock import Mock, patch


# Mock OpenAI to avoid actual API calls
class MockOpenAIResponse:
    def __init__(self):
        self.choices = [Mock(message=Mock(content="Test response"))]
        self.model = "gpt-4"
        self.usage = Mock(total_tokens=100)


class MockOpenAIClient:
    def __init__(self, **kwargs):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock(return_value=MockOpenAIResponse())


def measure_overhead(
    func: Callable, args: tuple = (), kwargs: dict = None, iterations: int = 1000
) -> List[float]:
    """Measure execution time for multiple iterations"""
    if kwargs is None:
        kwargs = {}

    times = []
    # Warm up
    for _ in range(10):
        func(*args, **kwargs)

    # Actual measurements
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return times


def calculate_percentiles(times: List[float]) -> Dict[str, float]:
    """Calculate P50, P95, P99 percentiles"""
    sorted_times = sorted(times)
    n = len(sorted_times)

    return {
        "min": min(sorted_times),
        "max": max(sorted_times),
        "mean": statistics.mean(sorted_times),
        "median": statistics.median(sorted_times),
        "p50": sorted_times[int(n * 0.50)],
        "p95": sorted_times[int(n * 0.95)],
        "p99": sorted_times[int(n * 0.99)],
        "std_dev": statistics.stdev(sorted_times) if n > 1 else 0,
    }


def test_baseline_performance():
    """Test baseline performance without Arc"""
    print("\n=== BASELINE PERFORMANCE (No Arc) ===")

    client = MockOpenAIClient()

    # Test different request sizes
    test_configs = [
        {"messages": [{"role": "user", "content": "Hi"}], "model": "gpt-4"},
        {
            "messages": [{"role": "user", "content": "Hi" * 100}],
            "model": "gpt-4",
            "temperature": 0.7,
        },
        {
            "messages": [{"role": "user", "content": "Hi" * 1000}],
            "model": "gpt-4",
            "temperature": 0.9,
            "max_tokens": 1000,
        },
    ]

    for i, config in enumerate(test_configs):
        times = measure_overhead(
            client.chat.completions.create, kwargs=config, iterations=1000
        )

        stats = calculate_percentiles(times)
        print(f"\nConfig {i+1} (params: {len(config)})")
        print(f"  Mean: {stats['mean']:.3f}ms")
        print(f"  P50:  {stats['p50']:.3f}ms")
        print(f"  P95:  {stats['p95']:.3f}ms")
        print(f"  P99:  {stats['p99']:.3f}ms")

    return times


def test_arc_interception_overhead():
    """Test Arc interception overhead"""
    print("\n=== ARC INTERCEPTION OVERHEAD ===")

    # Import Arc after mock is set up
    with patch.dict(
        "sys.modules",
        {"openai": Mock(OpenAI=MockOpenAIClient, AsyncOpenAI=MockOpenAIClient)},
    ):
        from runtime import Arc

        # Initialize Arc
        arc = Arc()

        # Create client that will be intercepted
        client = MockOpenAIClient()

        # Manually patch the client to simulate Arc interception
        arc.interceptors["openai"]._patch_completions(client.chat.completions)

        # Test configurations with varying complexity
        test_configs = [
            {
                "name": "Simple request (no pattern match)",
                "config": {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "model": "gpt-3.5-turbo",
                },
                "should_match": False,
            },
            {
                "name": "Request with pattern match",
                "config": {
                    "messages": [{"role": "user", "content": "Hi"}],
                    "model": "gpt-4",
                    "temperature": 0.95,
                },
                "should_match": True,
            },
            {
                "name": "Large request with pattern match",
                "config": {
                    "messages": [{"role": "user", "content": "Hi" * 1000}],
                    "model": "gpt-4",
                    "temperature": 0.95,
                    "max_tokens": 2000,
                },
                "should_match": True,
            },
        ]

        # Register a pattern that will match some requests
        arc.register_pattern(
            pattern={"model": "gpt-4", "temperature": {">": 0.9}},
            fix={"temperature": 0.7},
        )

        overhead_times = []

        for test in test_configs:
            print(f"\n{test['name']}:")

            # Measure with Arc
            arc_times = measure_overhead(
                client.chat.completions.create, kwargs=test["config"], iterations=1000
            )

            # Measure baseline (mock the original call directly)
            baseline_mock = Mock(return_value=MockOpenAIResponse())
            baseline_times = measure_overhead(
                baseline_mock, kwargs=test["config"], iterations=1000
            )

            # Calculate overhead
            overhead = [arc - base for arc, base in zip(arc_times, baseline_times)]
            overhead_stats = calculate_percentiles(overhead)

            overhead_times.extend(overhead)

            print(f"  Arc overhead:")
            print(f"    Mean: {overhead_stats['mean']:.3f}ms")
            print(f"    P50:  {overhead_stats['p50']:.3f}ms")
            print(f"    P95:  {overhead_stats['p95']:.3f}ms")
            print(f"    P99:  {overhead_stats['p99']:.3f}ms")
            print(f"    Max:  {overhead_stats['max']:.3f}ms")

            # Check PRD requirement
            if overhead_stats["p99"] < 5.0:
                print(
                    f"    ✓ PASS: P99 overhead ({overhead_stats['p99']:.3f}ms) < 5ms requirement"
                )
            else:
                print(
                    f"    ✗ FAIL: P99 overhead ({overhead_stats['p99']:.3f}ms) exceeds 5ms requirement"
                )

        # Overall overhead statistics
        overall_stats = calculate_percentiles(overhead_times)
        print("\n=== OVERALL ARC OVERHEAD ===")
        print(f"Total samples: {len(overhead_times)}")
        print(f"Mean: {overall_stats['mean']:.3f}ms")
        print(f"P50:  {overall_stats['p50']:.3f}ms")
        print(f"P95:  {overall_stats['p95']:.3f}ms")
        print(f"P99:  {overall_stats['p99']:.3f}ms")
        print(f"Max:  {overall_stats['max']:.3f}ms")

        if overall_stats["p99"] < 5.0:
            print(
                f"\n✅ PASS: Overall P99 overhead ({overall_stats['p99']:.3f}ms) meets <5ms requirement"
            )
        else:
            print(
                f"\n❌ FAIL: Overall P99 overhead ({overall_stats['p99']:.3f}ms) exceeds 5ms requirement"
            )

        return overall_stats


def test_concurrent_performance():
    """Test Arc performance under concurrent load"""
    print("\n=== CONCURRENT PERFORMANCE TEST ===")

    with patch.dict(
        "sys.modules",
        {"openai": Mock(OpenAI=MockOpenAIClient, AsyncOpenAI=MockOpenAIClient)},
    ):
        from runtime import Arc

        # Initialize Arc
        arc = Arc()

        # Register pattern
        arc.register_pattern(
            pattern={"model": "gpt-4", "temperature": {">": 0.9}},
            fix={"temperature": 0.7},
        )

        def make_request():
            client = MockOpenAIClient()
            arc.interceptors["openai"]._patch_completions(client.chat.completions)

            start = time.perf_counter()
            client.chat.completions.create(
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4",
                temperature=0.95,
            )
            end = time.perf_counter()
            return (end - start) * 1000

        # Test with different thread counts
        thread_counts = [1, 5, 10, 20]

        for num_threads in thread_counts:
            print(f"\nTesting with {num_threads} concurrent threads:")

            all_times = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                # Submit 1000 requests across threads
                futures = [executor.submit(make_request) for _ in range(1000)]

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    all_times.append(future.result())

            stats = calculate_percentiles(all_times)
            print(f"  Mean: {stats['mean']:.3f}ms")
            print(f"  P50:  {stats['p50']:.3f}ms")
            print(f"  P95:  {stats['p95']:.3f}ms")
            print(f"  P99:  {stats['p99']:.3f}ms")

            if stats["p99"] < 5.0:
                print(
                    f"  ✓ PASS: P99 ({stats['p99']:.3f}ms) < 5ms under {num_threads} threads"
                )
            else:
                print(
                    f"  ✗ FAIL: P99 ({stats['p99']:.3f}ms) exceeds 5ms under {num_threads} threads"
                )


def generate_performance_report(results: Dict[str, Any]):
    """Generate a detailed performance report"""
    report = f"""
# Arc Runtime Performance Test Report

## Executive Summary
- **P99 Latency Requirement**: <5ms
- **Actual P99 Latency**: {results['p99']:.3f}ms
- **Status**: {'✅ PASS' if results['p99'] < 5.0 else '❌ FAIL'}

## Detailed Metrics
- **Mean Overhead**: {results['mean']:.3f}ms
- **Median (P50)**: {results['p50']:.3f}ms
- **P95**: {results['p95']:.3f}ms
- **P99**: {results['p99']:.3f}ms
- **Max**: {results['max']:.3f}ms
- **Std Dev**: {results['std_dev']:.3f}ms

## Test Configuration
- **Iterations**: 1000 per test case
- **Test Cases**: 
  - Simple requests without pattern matching
  - Requests with pattern matching
  - Large requests with complex parameters
- **Concurrent Load**: Tested with 1, 5, 10, and 20 threads

## Recommendations
"""

    if results["p99"] < 5.0:
        report += "- Arc Runtime meets the <5ms P99 latency requirement\n"
        report += "- Performance is acceptable for production use\n"
    else:
        report += "- Arc Runtime does NOT meet the <5ms P99 latency requirement\n"
        report += "- Performance optimization needed before production use\n"
        report += "- Consider profiling the interception logic for bottlenecks\n"

    return report


def main():
    """Run all performance tests"""
    print("=" * 80)
    print("ARC RUNTIME PERFORMANCE TEST SUITE")
    print("Testing <5ms P99 interception overhead requirement")
    print("=" * 80)

    # Run baseline test
    test_baseline_performance()

    # Run Arc overhead test
    overhead_stats = test_arc_interception_overhead()

    # Run concurrent test
    test_concurrent_performance()

    # Generate report
    report = generate_performance_report(overhead_stats)
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)
    print(report)

    # Save report to file
    with open("/Users/jarrodbarnes/runtime/performance_report.md", "w") as f:
        f.write(report)
    print("\nReport saved to: performance_report.md")

    # Return exit code based on pass/fail
    return 0 if overhead_stats["p99"] < 5.0 else 1


if __name__ == "__main__":
    exit(main())
