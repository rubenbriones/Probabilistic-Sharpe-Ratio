import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.sharpe_ratio_stats import estimated_sharpe_ratio, estimated_sharpe_ratio_stdev, \
    probabilistic_sharpe_ratio, min_track_record_length


class TestSharpeRatio(unittest.TestCase):

    def setUp(self):
        path = Path(__file__).parent
        self.returns = pd.read_csv(path / 'returns_for_tests.csv', header=None, squeeze=True)

    def test_estimated_sharpe_ratio(self):
        EXPECTED_SR = 0.24171729852769172
        
        # pd.Series
        sr = estimated_sharpe_ratio(self.returns)
        self.assertAlmostEqual(sr, EXPECTED_SR)

        # pd.DataFrame
        returns_df = self.returns.to_frame()
        sr = estimated_sharpe_ratio(returns_df)
        pd.testing.assert_series_equal(sr, pd.Series(EXPECTED_SR, index=returns_df.columns))

        # np.array
        sr = estimated_sharpe_ratio(self.returns.values)
        self.assertAlmostEqual(sr, EXPECTED_SR)

    def test_estimated_sharpe_ratio_stdev(self):
        EXPECTED_SR_STD = 0.16466174867277414

        # pd.Series
        sr_std = estimated_sharpe_ratio_stdev(self.returns)
        self.assertAlmostEqual(sr_std, EXPECTED_SR_STD)

        # pd.DataFrame
        returns_df = self.returns.to_frame()
        sr_std = estimated_sharpe_ratio_stdev(returns_df)
        pd.testing.assert_series_equal(sr_std, pd.Series(EXPECTED_SR_STD, index=returns_df.columns))

        # np.array
        sr_std = estimated_sharpe_ratio_stdev(self.returns.values)
        self.assertAlmostEqual(sr_std, EXPECTED_SR_STD)

    def test_estimated_sharpe_ratio_stdev_paper_examples(self):
        sr_std = estimated_sharpe_ratio_stdev(n=134, skew=-0.2250, kurtosis=2.9570, sr=0.7079)
        self.assertAlmostEqual(round(sr_std, 4), 0.1028)

        sr_std = estimated_sharpe_ratio_stdev(n=134, skew=-1.4455, kurtosis=7.0497, sr=0.8183)
        self.assertAlmostEqual(round(sr_std, 4), 0.1550)

    def test_probabilistic_sharpe_ratio(self):
        SR_BENCHMARK = 0.05
        EXPECTED_PSR = 0.877850770122146

        # pd.Series
        psr = probabilistic_sharpe_ratio(self.returns, SR_BENCHMARK)
        self.assertAlmostEqual(psr, EXPECTED_PSR)

        # pd.DataFrame
        returns_df = self.returns.to_frame()
        psr = probabilistic_sharpe_ratio(returns_df, SR_BENCHMARK)
        pd.testing.assert_series_equal(psr, pd.Series(EXPECTED_PSR, index=returns_df.columns))

        # np.array
        psr = probabilistic_sharpe_ratio(self.returns.values, SR_BENCHMARK)
        self.assertAlmostEqual(psr, EXPECTED_PSR)

    def test_probabilistic_sharpe_ratio_paper_examples(self):
        sr_std = estimated_sharpe_ratio_stdev(n=24, skew=0, kurtosis=3, sr=0.458)
        psr = probabilistic_sharpe_ratio(sr_benchmark=0, sr=0.458, sr_std=sr_std)
        self.assertAlmostEqual(round(psr, 3), 0.982)

        sr_std = estimated_sharpe_ratio_stdev(n=24, skew=-2.448, kurtosis=10.164, sr=0.458)
        psr = probabilistic_sharpe_ratio(sr_benchmark=0, sr=0.458, sr_std=sr_std)
        self.assertAlmostEqual(round(psr, 3), 0.913)

        psr = probabilistic_sharpe_ratio(sr_benchmark=0, sr=0.7079, sr_std=0.1028)
        self.assertAlmostEqual(round(psr, 4), 1.0)  # minTRL=0.7152 years

        psr = probabilistic_sharpe_ratio(sr_benchmark=0, sr=0.8183, sr_std=0.1550)
        self.assertAlmostEqual(round(psr, 4), 1.0)  # minTRL=1.1593 years

    def test_min_track_record_length(self):
        SR_BENCHMARK = 0.05
        PROB = 0.90
        EXPECTED_MIN_TRL = 60.365085269027084

        sr = estimated_sharpe_ratio(self.returns)
        sr_std = estimated_sharpe_ratio_stdev(self.returns)
        psr = probabilistic_sharpe_ratio(self.returns, SR_BENCHMARK)

        # pd.Series
        min_trl = min_track_record_length(self.returns, SR_BENCHMARK, PROB)
        self.assertAlmostEqual(min_trl, EXPECTED_MIN_TRL)

        # pd.DataFrame
        returns_df = self.returns.to_frame()
        min_trl = min_track_record_length(returns_df, SR_BENCHMARK, PROB)
        pd.testing.assert_series_equal(min_trl, pd.Series(EXPECTED_MIN_TRL, index=returns_df.columns))

        # np.array
        min_trl = min_track_record_length(self.returns.values, SR_BENCHMARK, PROB)
        self.assertAlmostEqual(min_trl, EXPECTED_MIN_TRL)
        
    def test_min_track_record_length_paper_examples(self):
        sr_std = estimated_sharpe_ratio_stdev(n=134, skew=-0.2250, kurtosis=2.9570, sr=0.7079)
        min_trl = min_track_record_length(sr_benchmark=0, prob=0.95, n=134, sr=0.7079, sr_std=sr_std)
        self.assertAlmostEqual(round(min_trl/12, 4), 0.7152)  # minTRL was in years in the paper (and SR was monthly)

        sr_std = estimated_sharpe_ratio_stdev(n=134, skew=-1.4455, kurtosis=7.0497, sr=0.8183)
        min_trl = min_track_record_length(sr_benchmark=0, prob=0.95, n=134, sr=0.8183, sr_std=sr_std)
        self.assertAlmostEqual(round(min_trl/12, 4), 1.1593)  # minTRL was in years in the paper (and SR was monthly)
