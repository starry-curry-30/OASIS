import csv
import logging
import time
from contextlib import contextmanager

import numpy as np
from pympler import asizeof
from tabulate import tabulate

# Set up the module-level logger
logger = logging.getLogger(__name__)

"""
A context manager that measures the time taken for a block of code to execute.

Parameters:
description (str): A description of the block of code being measured. Default is an empty string.

Yields:
None

Returns:
None

Example:
>>> with measure_time_block("My code block"):
>>>     # Code to be measured
>>>     pass
My code block: 0.01 seconds
"""


@contextmanager
def measure_time_block(description=""):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        message = f"{description}: {elapsed_time:.4f} seconds"

        # Check if logger is configured
        if logging.getLogger().hasHandlers():
            logger.info(message)
        else:
            print(message)


def measure_time(func):
    """
    A decorator function that measures the execution time of another function.

    Parameters:
    func (function): The function to be measured.

    Returns:
    function: The decorated function that measures the execution time.

    Example:
    >>> @measure_time
    >>> def my_function():
    >>>     # Code to be measured
    >>>     pass
    >>> my_function()
    Done! Execution time of my_function function: 0.01 seconds
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        # Call the function with any arguments it was called with
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper


@contextmanager
def timing_context(name, instance, results_attr_name="timing_results"):
    """
    Context manager for measuring execution time with optional storage.

    Args:
        name (str): Name of the block being measured.
        instance (object): Object to store timing results.
        results_attr_name (str): Name of the attribute to store timing results on the instance.
    """
    # initialize the dict
    if not hasattr(instance, results_attr_name):
        setattr(instance, results_attr_name, {})

    # get dict
    results_dict = getattr(instance, results_attr_name)

    # initialize the list with the given name
    if name not in results_dict:
        results_dict[name] = []

    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time

    # store the elapsed time in the dict
    results_dict[name].append(elapsed_time)


def mean_per_step_time(
    timing_results,
    step_name: str = "Time Per Frame",
):
    """
    Mean duration and count for one timing step (e.g. average time per frame).

    Returns:
        (mean_seconds, count), or (None, 0) if empty.
    """
    if not timing_results:
        return None, 0
    times = timing_results.get(step_name)
    if not times:
        return None, 0
    return float(np.mean(times)), int(len(times))


def keyframe_timing_memory_cluster_breakdown(
    timing_results,
    use_parallel: bool = False,
    frame_step_name: str = "Time Per Frame",
    memory_cluster_step_name: str = "Memory Cluster",
):
    """
    Summarize keyframe timing: total avg per frame, memory-cluster share, residual.

    Sequential (use_parallel=False): "Time Per Frame" is end-to-end; Memory Cluster is a sub-timer.
    If there are k Memory Cluster entries per keyframe on average,
    cluster time per keyframe ~= sum(Memory Cluster) / n_keyframes,
    residual ~= mean(Time Per Frame) - that.

    Parallel mode: main and mapping threads overlap; do not subtract to get wall time per frame (see note).
    """
    if not timing_results:
        return None
    pf = timing_results.get(frame_step_name) or []
    n = len(pf)
    if n == 0:
        return None
    avg_total = float(np.mean(pf))
    out = {
        "n_keyframes": n,
        "avg_time_per_frame_total": avg_total,
        "avg_time_per_frame_excl_memory_cluster": None,
        "memory_cluster_time_per_keyframe": None,
        "memory_cluster_invocations_per_keyframe": None,
        "mean_memory_cluster_invocation": None,
        "note": None,
    }
    mc = timing_results.get(memory_cluster_step_name) or []
    if use_parallel:
        out["note"] = (
            "Parallel mode: 'Time Per Frame' is main thread only (detection/enqueue, etc.); "
            "'Memory Cluster' runs on the mapping thread. They overlap; do not subtract for wall time. "
            "To compare without clustering/rep frames, use use_parallel=false or disable memory_cluster.enabled."
        )
        if mc:
            out["mean_memory_cluster_invocation"] = float(np.mean(mc))
            out["memory_cluster_time_per_keyframe"] = None
        return out
    if not mc:
        out["avg_time_per_frame_excl_memory_cluster"] = avg_total
        out["note"] = (
            "No Memory Cluster samples (e.g. memory_cluster.enabled=false); "
            "per-frame total equals average without cluster subtraction."
        )
        return out
    n_mc = len(mc)
    if n_mc % n != 0:
        out["note"] = (
            f"Memory Cluster count ({n_mc}) not divisible by keyframe count ({n}); "
            "cannot allocate per-keyframe cluster time reliably."
        )
        out["mean_memory_cluster_invocation"] = float(np.mean(mc))
        return out
    k = n_mc // n
    cluster_per_kf = float(np.sum(mc)) / n
    out["memory_cluster_invocations_per_keyframe"] = k
    out["memory_cluster_time_per_keyframe"] = cluster_per_kf
    out["mean_memory_cluster_invocation"] = float(np.mean(mc))
    out["avg_time_per_frame_excl_memory_cluster"] = avg_total - cluster_per_kf
    return out


def print_timing_results(label, timing_results):
    if not timing_results:
        return
    rows = []
    for key, times in timing_results.items():
        avg_time = np.mean(times)
        percentile_90 = np.percentile(times, 90)
        rows.append([key, f"{avg_time:.4f}", f"{percentile_90:.4f}"])
    logger.info(f"\n{label} Timing Results:")
    logger.info(
        tabulate(
            rows,
            headers=["Step", "Avg Time (s)", "90th Percentile (s)"],
            tablefmt="grid",
        )
    )


def save_timing_results(timing_results, csv_file):
    if not timing_results:
        return

    rows = []
    for key, times in timing_results.items():
        avg_time = np.mean(times)
        percentile_90 = np.percentile(times, 90)
        rows.append([key, f"{avg_time:.4f}", f"{percentile_90:.4f}"])

    # Save results to CSV if a filename is provided
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Avg Time (s)", "90th Percentile (s)"])
        for row in rows:
            writer.writerow(row)
    logger.info(f"\nTiming results saved to {csv_file}")


# Utility function to get memory usage of local and global maps
def get_map_memory_usage(local_map, global_map):
    local_mb = asizeof.asizeof(local_map) / 1024 / 1024
    global_mb = asizeof.asizeof(global_map) / 1024 / 1024
    return {"local_map_mb": round(local_mb, 4), "global_map_mb": round(global_mb, 4)}
