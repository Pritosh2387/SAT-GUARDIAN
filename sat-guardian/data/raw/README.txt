This directory contains raw INSAT-3DS satellite data files (.nc format).

Place your files here:
- insat3ds_T0.nc     – First observation frame
- insat3ds_T1.nc     – Second observation frame (30 min later)

Variables expected:
- brightness_temperature  (H x W float array)

If no real data is available, run `python scripts/demo.py` which generates
synthetic sample data automatically.
