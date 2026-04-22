"""
carbon_tracker.py — Carbon Emissions Tracking for VulnHGNN.

Uses CodeCarbon to measure energy consumption and CO₂ emissions across:
  - Training phase
  - Inference phase
  - Repair and validation pipeline

Metrics tracked:
  - Energy consumption (kWh)
  - CO₂ emissions (kg)
  - Duration (seconds)

Purpose:
  - Compare efficiency with baseline methods
  - Evaluate trade-off between accuracy and energy usage
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("carbon_tracker")


class CarbonTracker:
    """
    Wraps CodeCarbon's EmissionsTracker for tracking energy and CO₂
    across different pipeline phases.

    Usage:
        tracker = CarbonTracker(project_name="VulnHGNN")

        # Track training
        with tracker.track("training"):
            train_model(...)

        # Track inference
        with tracker.track("inference"):
            predict(...)

        # Get combined report
        report = tracker.get_report()
    """

    def __init__(
        self,
        project_name: str = "VulnHGNN",
        output_dir: str = "results",
        log_level: str = "warning",
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = log_level

        self._tracker = None
        self._codecarbon_available = False
        self._phase_metrics = {}
        self._current_phase = None

        # Try to import codecarbon
        try:
            from codecarbon import EmissionsTracker
            self._codecarbon_available = True
            logger.info("CodeCarbon available — emissions tracking enabled.")
        except ImportError:
            logger.warning(
                "CodeCarbon not installed. Install with: pip install codecarbon. "
                "Carbon tracking will use fallback (time-only) mode."
            )

    def track(self, phase_name: str):
        """
        Context manager for tracking a specific phase.

        Usage:
            with tracker.track("training"):
                train_model(...)
        """
        return _PhaseTracker(self, phase_name)

    def start_phase(self, phase_name: str):
        """Start tracking a named phase."""
        self._current_phase = phase_name

        if self._codecarbon_available:
            try:
                from codecarbon import EmissionsTracker
                self._tracker = EmissionsTracker(
                    project_name=f"{self.project_name}_{phase_name}",
                    output_dir=str(self.output_dir),
                    log_level=self.log_level,
                    save_to_file=True,
                )
                self._tracker.start()
            except Exception as e:
                logger.warning("CodeCarbon start failed: %s — using fallback", e)
                self._tracker = None

        self._phase_metrics[phase_name] = {
            "phase": phase_name,
            "start_time": time.time(),
            "end_time": None,
            "duration_sec": 0,
            "energy_kwh": 0,
            "co2_kg": 0,
            "tracked_by": "codecarbon" if self._tracker else "fallback",
        }

    def stop_phase(self):
        """Stop tracking the current phase and record metrics."""
        if not self._current_phase:
            return

        phase = self._current_phase
        metrics = self._phase_metrics.get(phase, {})
        metrics["end_time"] = time.time()
        metrics["duration_sec"] = round(
            metrics["end_time"] - metrics["start_time"], 3
        )

        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
                if emissions is not None:
                    metrics["co2_kg"] = round(float(emissions), 8)

                # Read detailed data from CodeCarbon output
                emissions_file = self.output_dir / "emissions.csv"
                if emissions_file.exists():
                    metrics["energy_kwh"] = self._extract_energy(emissions_file)

            except Exception as e:
                logger.warning("CodeCarbon stop failed: %s", e)

        self._phase_metrics[phase] = metrics
        self._tracker = None
        self._current_phase = None

        logger.info(
            "Phase '%s' tracked: duration=%.2fs, energy=%.6f kWh, CO₂=%.8f kg",
            phase,
            metrics["duration_sec"],
            metrics["energy_kwh"],
            metrics["co2_kg"],
        )

    def _extract_energy(self, csv_path: Path) -> float:
        """Extract energy consumption from CodeCarbon CSV output."""
        try:
            import csv
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    return round(float(last.get("energy_consumed", 0)), 8)
        except Exception as e:
            logger.warning("Could not parse CodeCarbon CSV: %s", e)
        return 0.0

    def get_report(self) -> dict:
        """
        Get combined carbon emissions report across all tracked phases.

        Returns:
            dict with per-phase metrics and totals
        """
        total_duration = 0
        total_energy = 0
        total_co2 = 0

        for metrics in self._phase_metrics.values():
            total_duration += metrics.get("duration_sec", 0)
            total_energy += metrics.get("energy_kwh", 0)
            total_co2 += metrics.get("co2_kg", 0)

        report = {
            "project": self.project_name,
            "phases": dict(self._phase_metrics),
            "totals": {
                "duration_sec": round(total_duration, 3),
                "energy_kwh": round(total_energy, 8),
                "co2_kg": round(total_co2, 8),
            },
            "codecarbon_available": self._codecarbon_available,
        }

        return report

    def save_report(self, filepath: Optional[str] = None):
        """Save carbon report to JSON file."""
        if filepath is None:
            filepath = str(self.output_dir / "carbon_report.json")

        report = self.get_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Carbon report saved → %s", filepath)
        return filepath

    def print_report(self):
        """Print a formatted carbon report to console."""
        report = self.get_report()
        sep = "─" * 60

        print(f"\n{sep}")
        print(f"  CARBON EMISSIONS REPORT — {self.project_name}")
        print(sep)

        if not report["codecarbon_available"]:
            print("  ⚠ CodeCarbon not installed — showing time-only metrics")
            print()

        print(f"  {'Phase':<20} {'Duration':<12} {'Energy (kWh)':<14} {'CO₂ (kg)':<12}")
        print(f"  {'-'*56}")

        for phase, metrics in report["phases"].items():
            print(
                f"  {phase:<20} {metrics['duration_sec']:>8.2f}s   "
                f"{metrics['energy_kwh']:>12.6f}   "
                f"{metrics['co2_kg']:>10.8f}"
            )

        totals = report["totals"]
        print(f"  {'-'*56}")
        print(
            f"  {'TOTAL':<20} {totals['duration_sec']:>8.2f}s   "
            f"{totals['energy_kwh']:>12.6f}   "
            f"{totals['co2_kg']:>10.8f}"
        )
        print(sep)


class _PhaseTracker:
    """Context manager for tracking a single phase."""

    def __init__(self, tracker: CarbonTracker, phase_name: str):
        self.tracker = tracker
        self.phase_name = phase_name

    def __enter__(self):
        self.tracker.start_phase(self.phase_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop_phase()
        return False


# ════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def track_function(
    func: Callable,
    phase_name: str,
    project_name: str = "VulnHGNN",
    *args,
    **kwargs,
) -> tuple:
    """
    Convenience function to track carbon emissions for a single function call.

    Returns:
        (function_result, carbon_metrics_dict)
    """
    tracker = CarbonTracker(project_name=project_name)
    with tracker.track(phase_name):
        result = func(*args, **kwargs)
    return result, tracker.get_report()
