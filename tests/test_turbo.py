import subprocess
import unittest
from dataclasses import replace
from unittest.mock import patch

from kerdoios.providers.http import JsonResponse
from kerdoios.quota import QuotaDimension, QuotaState
from kerdoios.types import Mode
from kerdoios.turbo import (
    DEFAULT_ADMISSION, OPENCODE_MODEL, choose_turbo_offer, discover_turbo_offers,
)


def catalog(url, **kwargs):
    model = 'llama-3.1-8b-instant' if 'groq.com' in url else 'gpt-oss-120b'
    return JsonResponse({'data': [{'id': model}, {'id': 'paid-only'}]}, {})


class TurboTests(unittest.TestCase):
    def inventory(self, output=OPENCODE_MODEL + '\n'):
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test', 'CEREBRAS_API_KEY': 'test'}, clear=True), \
             patch('kerdoios.providers.openai_compat.get_json_response', side_effect=catalog), \
             patch('kerdoios.turbo.subprocess.run', return_value=subprocess.CompletedProcess([], 0, output, '')) as run:
            offers = discover_turbo_offers()
        self.assertEqual(run.call_args.args[0], ['/usr/bin/opencode', 'models'])
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertNotIn('shell', run.call_args.kwargs)
        return offers

    def test_inventory_is_live_free_tier_only_and_caps_are_client_policy(self):
        offers = self.inventory()
        self.assertEqual({o.provider for o in offers}, {'groq', 'cerebras', 'opencode'})
        self.assertEqual(DEFAULT_ADMISSION['groq']['concurrency'], 1)
        self.assertEqual(DEFAULT_ADMISSION['groq']['rpm'], 6)
        self.assertIsNone(DEFAULT_ADMISSION['groq']['tokens_per_minute'])
        for offer in offers:
            self.assertEqual(offer.capacity.concurrency, 1)
            self.assertEqual(offer.capacity.requests_per_minute, 6)
            self.assertIsNone(offer.capacity.tokens_per_minute)
            self.assertNotEqual(offer.source, 'fixture')
            self.assertIsNone(offer.economics.quota.dim('tpd').remaining)
            self.assertIsNone(offer.telemetry.output_tok_s)
            if offer.provider != 'opencode':
                self.assertIsNone(offer.economics.input_token_price)
                self.assertIsNone(offer.economics.output_token_price)

    def test_opencode_absent_or_substring_is_not_an_offer(self):
        for output in ('', OPENCODE_MODEL + '-other\n', 'error: ' + OPENCODE_MODEL):
            self.assertEqual(len(self.inventory(output)), 2)

    def test_no_keys_no_network_but_opencode_needs_no_key(self):
        with patch.dict('os.environ', {}, clear=True), \
             patch('kerdoios.providers.openai_compat.get_json_response') as network, \
             patch('kerdoios.turbo.subprocess.run', return_value=subprocess.CompletedProcess([], 0, OPENCODE_MODEL + '\n', '')):
            offers = discover_turbo_offers()
        network.assert_not_called()
        self.assertEqual([o.id for o in offers], [OPENCODE_MODEL])

    def test_catalog_failure_and_cli_failure_return_no_fixture(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('opencode', 20), subprocess.CalledProcessError(1, 'opencode')):
            with patch.dict('os.environ', {'GROQ_API_KEY': 'test', 'CEREBRAS_API_KEY': 'test'}, clear=True), \
                 patch('kerdoios.providers.openai_compat.get_json_response', return_value=None), \
                 patch('kerdoios.turbo.subprocess.run', side_effect=error):
                self.assertEqual(discover_turbo_offers(), [])

    def test_routing_exclusions_and_original_object(self):
        offers = self.inventory()
        self.assertIs(choose_turbo_offer(offers, coding=True), offers[2])
        self.assertIsNone(choose_turbo_offer(offers, coding=True, excluded=('opencode',)))
        self.assertIsNone(choose_turbo_offer(offers, excluded=('groq', 'cerebras')))
        self.assertIsNone(choose_turbo_offer([]))
        self.assertEqual(choose_turbo_offer(offers, excluded=('groq',)).provider, 'cerebras')
        self.assertIsNone(choose_turbo_offer([replace(offers[0], source='fixture')]))
        self.assertIsNone(choose_turbo_offer([replace(offers[0], model='paid-only')]))

    def test_measurements_drive_fast_plan_without_mutating_offers(self):
        offers = self.inventory()[:2]
        for field, a, b in [('latency_ms', 5000, 10), ('output_tok_s', 1, 1000)]:
            measurements = {offers[0].id: {field: a}, offers[1].id: {field: b}}
            self.assertIs(choose_turbo_offer(offers, measurements=measurements), offers[1])
            measurements = {offers[0].id: {field: b}, offers[1].id: {field: a}}
            self.assertIs(choose_turbo_offer(offers, measurements=measurements), offers[0])
        self.assertIsNone(offers[0].telemetry.output_tok_s)
        with patch('kerdoios.turbo.plan', wraps=__import__('kerdoios.optimize', fromlist=['plan']).plan) as planner:
            choose_turbo_offer(offers, measurements={offers[0].id: {'latency_ms': None, 'output_tok_s': 0}})
        projected, requirement = planner.call_args.args
        self.assertEqual(requirement.mode, Mode.FAST)
        self.assertGreater(projected[0].telemetry.latency_p50_ms, 0)
        self.assertIsNone(projected[0].telemetry.output_tok_s)

    def test_explicit_new_free_catalog_row_remains_eligible(self):
        response = JsonResponse({'data': [{'id': 'new-chat', 'free_tier': True}]}, {})
        with patch.dict('os.environ', {'GROQ_API_KEY': 'test'}, clear=True), \
             patch('kerdoios.providers.openai_compat.get_json_response', return_value=response), \
             patch('kerdoios.turbo.subprocess.run', side_effect=FileNotFoundError()):
            offers = discover_turbo_offers()
        self.assertEqual(len(offers), 1)
        self.assertIs(choose_turbo_offer(offers), offers[0])
        self.assertIsNone(offers[0].economics.input_token_price)

    def test_invalid_observations_do_not_become_zero(self):
        offers = self.inventory()[:2]
        for value in (None, 0, -1, float('nan'), float('inf'), True, 'bad'):
            with patch('kerdoios.turbo.plan', wraps=__import__('kerdoios.optimize', fromlist=['plan']).plan) as planner:
                choose_turbo_offer(offers, measurements={offers[0].id: {'latency_ms': value, 'output_tok_s': value}})
            telemetry = planner.call_args.args[0][0].telemetry
            self.assertEqual(telemetry.latency_p50_ms, offers[0].telemetry.latency_p50_ms)
            self.assertIsNone(telemetry.output_tok_s)

    def test_known_exhausted_window_is_excluded_unknown_is_not_fabricated(self):
        offer = self.inventory()[0]
        spent = QuotaState(provider='groq', dimensions={'tpd': QuotaDimension(remaining=0, source='provider_header')})
        self.assertIsNone(choose_turbo_offer([replace(offer, economics=replace(offer.economics, quota=spent))]))
        self.assertIs(choose_turbo_offer([offer]), offer)


if __name__ == '__main__':
    unittest.main()
