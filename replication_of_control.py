"""
This example reproduces results in Figure 6.7, where ramp metering control and
variable speed limits are controlled in the coordinated fashion. The control is achieved via
a Model Predictive Control (MPC) scheme, which is implemented with the CasADI-based 'csnlp' library.
"""

# importing all needed packages

# casadi solves nonlinear optimization problems, e.g. MPC
import casadi as cs

# visualization library
import matplotlib.pyplot as plt

# library for numerical computations e.g. arrays and matrices
import numpy as np

# core optimization framework
from csnlp import Nlp

# wrapper for MPC
from csnlp.wrappers import Mpc

# symbolic implementation of METANET
import sym_metanet as metanet
from sym_metanet import (
    Destination, # location where vehicles exit the freeway
    Link, # a freeway segment
    LinkWithVsl, # a freeway segment with variable speed limit
    MainstreamOrigin, # entry point of traffic onto the freeway
    MeteredOnRamp, # on ramp with ramp metering control
    Network, # combines all links, nodes, and control mechanisms
    Node, # intersection of freeway junctions
    engines, # simulation engines used for running the traffic model
)

# function creating a demand profile for vehicles entering the freeway.
def create_demands(time: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.interp(time, (2.0, 2.25), (3500, 1000)),
            np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500)),
        )
    )




