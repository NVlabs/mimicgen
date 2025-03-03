# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of miscellaneous utilities.
"""
import time
import json
import numpy as np

from collections import deque, OrderedDict
from contextlib import contextmanager


def add_red_border_to_frame(frame, ratio=0.02):
    """
    Add a red border to an image frame.
    """
    border_size_x = max(1, round(ratio * frame.shape[0]))
    border_size_y = max(1, round(ratio * frame.shape[1]))

    frame[:border_size_x, :, :] = [255., 0., 0.]
    frame[-border_size_x:, :, :] = [255., 0., 0.]
    frame[:, :border_size_y, :] = [255., 0., 0.]
    frame[:, -border_size_y:, :] = [255., 0., 0.]
    return frame


def deep_update(d, u):
    """
    Recursively update a mapping.
    """
    import collections
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Grid(object):
    """
    Keep track of a list of values, and point to a single value at a time.
    """
    def __init__(self, values, initial_ind=0):
        self.values = list(values)
        self.ind = initial_ind
        self.n = len(self.values)

    def get(self):
        return self.values[self.ind]

    def next(self):
        self.ind = min(self.ind + 1, self.n - 1)
        return self.get()

    def prev(self):
        self.ind = max(self.ind - 1, 0)
        return self.get()


class Timer(object):
    """
    A simple timer.
    """
    def __init__(self, history=100, ignore_first=False):
        """
        Args:
            history (int): number of recent timesteps to record for reporting statistics
        """
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.last_diff = 0.
        self.average_time = 0.
        self.min_diff = float("inf")
        self.max_diff = 0.
        self._measurements = deque(maxlen=history)
        self._enabled = True
        self.ignore_first = ignore_first
        self._had_first = False

    def enable(self):
        """
        Enable measurements with this timer.
        """
        self._enabled = True

    def disable(self):
        """
        Disable measurements with this timer.
        """
        self._enabled = False

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        if self._enabled:

            if self.ignore_first and (self.start_time > 0. and not self._had_first):
                self._had_first = True
                return time.time() - self.start_time

            self.last_diff = time.time() - self.start_time
            self.total_time += self.last_diff
            self.calls += 1
            self.average_time = self.total_time / self.calls
            self.min_diff = min(self.min_diff, self.last_diff)
            self.max_diff = max(self.max_diff, self.last_diff)
            self._measurements.append(self.last_diff)
        last_diff = self.last_diff
        return last_diff

    @contextmanager
    def timed(self):
        self.tic()
        yield
        self.toc()

    def report_stats(self, verbose=False):
        stats = OrderedDict()
        stats["global"] = OrderedDict(
            mean=self.average_time,
            min=self.min_diff,
            max=self.max_diff,
            num=self.calls,
        )
        num = len(self._measurements)
        stats["local"] = OrderedDict()
        if num > 0:
            stats["local"] = OrderedDict(
                mean=np.mean(self._measurements),
                std=np.std(self._measurements),
                min=np.min(self._measurements),
                max=np.max(self._measurements),
                num=num,
            )
        if verbose:
            stats["local"]["values"] = list(self._measurements)
        return stats


class Rate(object):
    """
    Convenience class for enforcing rates in loops. Modeled after rospy.Rate.

    See http://docs.ros.org/en/jade/api/rospy/html/rospy.timer-pysrc.html#Rate.sleep
    """
    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.update_hz(hz)

    def update_hz(self, hz):
        """
        Update rate to enforce.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = (1. / hz)

    def _remaining(self, curr_time):
        """
        Calculate time remaining for rate to sleep.
        """
        assert curr_time >= self.last_time, "time moved backwards!"
        elapsed = curr_time - self.last_time
        return self.sleep_duration - elapsed

    def sleep(self):
        """
        Attempt to sleep at the specified rate in hz, by taking the time
        elapsed since the last call to this function into account.
        """
        curr_time = time.time()
        remaining = self._remaining(curr_time)
        if remaining > 0:
            time.sleep(remaining)

        # assume successful rate sleeping
        self.last_time = self.last_time + self.sleep_duration

        # NOTE: this commented line is what we used to do, but this enforces a slower rate
        # self.last_time = time.time()

        # detect time jumping forwards (e.g. loop is too slow)
        if curr_time - self.last_time > self.sleep_duration * 2:
            # we didn't sleep at all
            self.last_time = curr_time


class RateMeasure(object):
    """
    Measure approximate time intervals of code execution by calling @measure
    """
    def __init__(self, name=None, history=100, freq_threshold=None):
        self._timer = Timer(history=history, ignore_first=True)
        self._timer.tic()
        self.name = name
        self.freq_threshold = freq_threshold
        self._enabled = True
        self._first = False
        self.sum = 0.
        self.calls = 0

    def enable(self):
        """
        Enable measurements.
        """
        self._timer.enable()
        self._enabled = True

    def disable(self):
        """
        Disable measurements.
        """
        self._timer.disable()
        self._enabled = False

    def measure(self):
        """
        Take a measurement of the time elapsed since the last @measure call
        and also return the time elapsed.
        """
        interval = self._timer.toc()
        self._timer.tic()
        self.sum += (1. / interval)
        self.calls += 1
        if self._enabled and (self.freq_threshold is not None) and ((1. / interval) < self.freq_threshold):
            print("WARNING: RateMeasure {} violated threshold {} hz with measurement {} hz".format(self.name, self.freq_threshold, (1. / interval)))
            return (interval, True)
        return (interval, False)

    def report_stats(self, verbose=False):
        """
        Report statistics over measurements, converting timer measurements into frequencies.
        """
        stats = self._timer.report_stats(verbose=verbose)
        stats["name"] = self.name
        if stats["global"]["num"] > 0:
            stats["global"] = OrderedDict(
                mean=(self.sum / float(self.calls)),
                min=(1. / stats["global"]["max"]),
                max=(1. / stats["global"]["min"]),
                num=stats["global"]["num"],
            )
        if len(stats["local"]) > 0:
            measurements = [1. / x for x in self._timer._measurements]
            stats["local"] = OrderedDict(
                mean=np.mean(measurements),
                std=np.std(measurements),
                min=np.min(measurements),
                max=np.max(measurements),
                num=stats["local"]["num"],
            )
        return stats

    def __str__(self):
        stats = self.report_stats(verbose=False)
        return json.dumps(stats, indent=4)
