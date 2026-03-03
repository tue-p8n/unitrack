r"""
UniTrack.

========

This module implements a tracker algorithm that maps detections to tracklets.

.. math::

    Tracker: Detections \rightarrow Tracklets

Each detection has fields that can be used to assign IDs from Tracklets in the
previous frame to Tracklets in the current frame.

Terminology
-----------

- **Detections**: All detected structures at the current frame.

- **Tracklets**: All detections from previous frames, each having a unique track ID.

- **Assignment**: The process that assigns each Detection to a Tracklet.

- **Lost**: The state of a Tracklet that has not been assigned to a detection at the
    current or a previous frame.
"""

__version__ = "1.0.1"

from . import assignment as assignment
from . import consts as consts
from . import costs as costs
from . import debug as debug
from . import stages as stages
from . import states as states
from ._memory import TrackletMemory, TrackletMemoryWriteReturnType
from ._tracker import MultiStageTracker, SelectField
from ._wrappers import SimpleTracker, StatefulTracker

__all__ = [
    "MultiStageTracker",
    "SelectField",
    "SimpleTracker",
    "StatefulTracker",
    "TrackletMemory",
    "TrackletMemoryWriteReturnType",
    "assignment",
    "consts",
    "costs",
    "debug",
    "stages",
    "states",
]
