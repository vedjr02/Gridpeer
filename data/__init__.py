"""Data loaders for GridPeer household demand and solar generation series.

Synthetic generators live here for week 1; the real CER (demand) and PVGIS
(solar) loaders land alongside them behind the same return shape, so swapping
synthetic for real data is a one-import change in the orchestrator.
"""
