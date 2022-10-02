#! /usr/bin/env python

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys
import pyrand
from pyrand import Memory
from pyrand import Timer


# ===========
# test device
# ===========

def test_device():
    """
    A test for pyrand.device.
    """

    # Device inquiry
    pyrand.info()
    pyrand.device.locate_cuda()
    pyrand.device.get_nvidia_driver_version()
    pyrand.device.get_processor_name()
    pyrand.device.get_gpu_name()
    pyrand.device.get_num_cpu_threads()
    pyrand.device.get_num_gpu_devices()
    pyrand.device.restrict_to_single_processor()

    # Memory
    mem = Memory()
    mem.start()
    mem.read()
    mem.read(human_readable=True)
    Memory.get_resident_memory()
    Memory.get_resident_memory(human_readable=True)

    # Timer
    timer = Timer(hold=True)
    timer.tic()
    timer.toc()
    timer.wall_time
    timer.proc_time


# ===========
# System Main
# ===========

if __name__ == "__main__":
    sys.exit(test_device())
