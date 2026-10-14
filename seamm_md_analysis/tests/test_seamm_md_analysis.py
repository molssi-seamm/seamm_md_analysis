"""
Unit and regression test for the seamm_md_analysis package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import seamm_md_analysis


def test_seamm_md_analysis_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "seamm_md_analysis" in sys.modules
