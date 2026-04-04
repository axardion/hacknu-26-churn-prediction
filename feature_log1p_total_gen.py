#!/usr/bin/env python3
"""
Append column `log1p_total_gen` to data/train/train_users.csv and data/test/test_users.csv.
Keeps all existing columns; only adds/updates this one.
"""

from add_features_generations_common import main_for_feature

if __name__ == "__main__":
    main_for_feature("log1p_total_gen")
