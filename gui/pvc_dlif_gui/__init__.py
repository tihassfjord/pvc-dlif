"""Tkinter front end for the pvc-dlif pipeline.

Three tabs, matching the three things you actually do:

* **PVC** - correct one image or a batch of them, with the parameters exposed.
* **Pipeline** - run the numbered stages 00-08 against a config file.
* **Analysis** - read the result tables and draw the plots.

Nothing here contains science.  Every button calls into ``pvc_dlif`` or shells
out to ``scripts/NN_*.py``, so the GUI and the command line always do the same
thing and there is only one implementation to trust.
"""

__version__ = "0.1.0"
