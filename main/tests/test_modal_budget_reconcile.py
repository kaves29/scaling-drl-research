import unittest
from datetime import datetime, timezone

from infra.modal.reconcile_failed_a100 import reconcile


class ReconcileTests(unittest.TestCase):
    def evidence(self):
        app = {'app_id': 'ap-ofE9iKlGWLLKdtZElwFuNr', 'state': 'stopped', 'tasks': '0',
               'created_at': '2026-11-17 03:46:13+00:00', 'stopped_at': '2026-11-17 03:48:27+00:00'}
        rows = [{'object_id': app['app_id'], 'interval_start': '2026-11-17T03:00:00',
                 'resource': name, 'cost': value, 'environment': 'main'} for name, value in
                [('A100-80GB', '0.06249994'), ('CPU', '0.00958809'), ('Memory', '0.01502165')]]
        return app, rows

    def test_adjusts_only_failed_reservation_and_keeps_evidence(self):
        app, rows = self.evidence()
        ledger = {'reserved_usd': 39.99957126, 'other': 'preserved'}
        result = reconcile(ledger, [app], rows, datetime(2026, 11, 17, 4, 1, tzinfo=timezone.utc),
                           {'ledger': 'a', 'apps': 'b', 'billing': 'c'})
        self.assertAlmostEqual(result['reserved_usd'], 32.01166296)
        self.assertEqual(result['other'], 'preserved')
        self.assertEqual(result['adjustments'][0]['app_id'], app['app_id'])
        self.assertEqual(ledger['reserved_usd'], 39.99957126)
        with self.assertRaises(ValueError):
            reconcile(result, [app], rows, datetime(2026, 11, 17, 4, 1, tzinfo=timezone.utc), {})

    def test_rejects_open_billing_or_unexpected_evidence(self):
        app, rows = self.evidence()
        ledger = {'reserved_usd': 39.99957126}
        with self.assertRaises(ValueError):
            reconcile(ledger, [app], rows, datetime(2026, 11, 17, 3, 59, tzinfo=timezone.utc), {})
        with self.assertRaises(ValueError):
            reconcile(ledger, [{**app, 'state': 'ephemeral'}], rows,
                      datetime(2026, 11, 17, 4, 1, tzinfo=timezone.utc), {})
        with self.assertRaises(ValueError):
            reconcile(ledger, [app], rows + [{**rows[0], 'resource': 'Network Egress', 'cost': '1.0'}],
                      datetime(2026, 11, 17, 4, 1, tzinfo=timezone.utc), {})
        with self.assertRaises(ValueError):
            reconcile({'reserved_usd': 39.99}, [app], rows,
                      datetime(2026, 11, 17, 4, 1, tzinfo=timezone.utc), {})
