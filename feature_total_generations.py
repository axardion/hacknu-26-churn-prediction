#!/usr/bin/env python3
"""
Append column `total_generations` to data/train/train_users.csv and data/test/test_users.csv.

All existing columns in those files are kept; this adds one column (or overwrites it if it
already exists). Same CLI flags as add_features_from_generations.py.
"""

from add_features_generations_common import main_for_feature

if __name__ == "__main__":
    main_for_feature("total_generations")
