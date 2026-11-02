"""E016 S2: regresja przedwczesnej weryfikacji i nadzoru procesu.

CPU, unittest; historyczny nieudany smoke jest fixture tylko do odczytu.
Nie generuje wyników naukowych. Protokół: OPIS_S2.md.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import launch_s2 as launcher
import run_s2_supervised as supervisor


class LifecycleTests(unittest.TestCase):
    def test_first_dataset_does_not_require_second(self):
        run = launcher.ROOT / 'runs/E016/s2_smoke_20261028T203618Z_cb9ffaf_bc9f84f7'
        cfg = launcher.common.read_json(run / 'config.json')
        manifest = launcher.common.read_json(run / 'manifest.json')
        report = launcher.verify_summaries(
            run, cfg, manifest['specs'], True, dataset_name='cifar10')
        self.assertEqual(set(report), {'cifar10'})
        with self.assertRaisesRegex(RuntimeError, 'cifar100'):
            launcher.verify_summaries(run, cfg, manifest['specs'], True)

    def test_supervisor_reports_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'worker.log'
            with patch.object(sys, 'argv', ['supervisor', '--log', str(log)]), \
                    patch.object(supervisor.subprocess, 'Popen') as start:
                start.return_value.pid = 123
                start.return_value.wait.return_value = 7
                self.assertEqual(supervisor.main(), 7)
                start.assert_called_once()
                state = launcher.common.read_json(log.with_suffix('.status.json'))
                self.assertEqual(state['status'], 'failed')

    def test_supervisor_timeout_stops_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'worker.log'
            with patch.object(sys, 'argv', ['supervisor', '--log', str(log)]), \
                    patch.object(supervisor.subprocess, 'Popen') as start, \
                    patch.object(supervisor, 'stop_tree') as stop:
                start.return_value.pid = 123
                start.return_value.wait.side_effect = subprocess.TimeoutExpired('test', 1)
                self.assertEqual(supervisor.main(), 124)
                stop.assert_called_once_with(start.return_value)
                self.assertEqual(launcher.common.read_json(
                    log.with_suffix('.status.json'))['status'], 'timeout')


if __name__ == '__main__':
    unittest.main()
