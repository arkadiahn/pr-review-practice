"""
Partition definitions for the Spotify pipeline.
Data is partitioned by month (e.g. "2024-01", "2024-02", ...).
"""

from dagster import MonthlyPartitionsDefinition

# Monthly partitions starting from when the dataset begins.
monthly_partitions = MonthlyPartitionsDefinition(start_date="2020-01-01")
