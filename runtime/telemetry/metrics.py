"""
Metrics collection for Arc Runtime
"""

import threading
import time
from collections import defaultdict
from typing import Dict, List


class Metrics:
    """
    Simple metrics collection for MVP
    Thread-safe counters and histograms
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._counters = defaultdict(int)
        self._histograms = defaultdict(list)
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1):
        """Increment a counter"""
        with self._lock:
            self._counters[name] += value

    def record_histogram(self, name: str, value: float):
        """Record a value in a histogram"""
        with self._lock:
            self._histograms[name].append(value)

    def get_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format

        Returns:
            Prometheus-formatted metrics string
        """
        with self._lock:
            lines = []

            # Add metadata
            lines.append(
                "# HELP arc_runtime_uptime_seconds Time since Arc Runtime started"
            )
            lines.append("# TYPE arc_runtime_uptime_seconds gauge")
            lines.append(f"arc_runtime_uptime_seconds {time.time() - self._start_time}")
            lines.append("")

            # Export counters
            for name, value in self._counters.items():
                safe_name = name.replace(".", "_").replace("-", "_")
                lines.append(f"# HELP {safe_name} Counter {name}")
                lines.append(f"# TYPE {safe_name} counter")
                lines.append(f"{safe_name} {value}")
                lines.append("")

            # Export histograms (simplified - just show count, sum, and avg)
            for name, values in self._histograms.items():
                if not values:
                    continue

                safe_name = name.replace(".", "_").replace("-", "_")
                count = len(values)
                total = sum(values)
                avg = total / count if count > 0 else 0

                lines.append(f"# HELP {safe_name} Histogram {name}")
                lines.append(f"# TYPE {safe_name} summary")
                lines.append(f"{safe_name}_count {count}")
                lines.append(f"{safe_name}_sum {total}")
                lines.append(
                    f'{safe_name}{{quantile="0.5"}} {avg}'
                )  # Simplified - just avg
                lines.append("")

            return "\n".join(lines)

    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
