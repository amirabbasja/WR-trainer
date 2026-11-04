import unittest

from workrunner.S3_protocol import KEXDialect, S3Job
from workrunner.S3_methods import s4_request
from workrunner.wire_constants import s6_noise_seed_prefix, s7_legacy_alias
from workrunner.wr_protocol import WRDialect, WRSubmission


def text_fixture(hex_bytes):
    return bytes.fromhex(hex_bytes).decode('ascii')


class WireCompatibilityTests(unittest.TestCase):
    def test_authentication_and_submission(self):
        dialect = KEXDialect('account', 'unit')
        calls = dialect.auth_requests()
        self.assertEqual([call['method'] for call in calls], [
            'compute.register', 'compute.session',
        ])
        self.assertEqual([s4_request(call)['method'] for call in calls], [
            'mining.subscribe', 'mining.authorize',
        ])
        self.assertEqual(calls[1]['params'], ['account.unit', 'x'])
        gzip_call, = KEXDialect('account', 'unit', request_gzip=True).auth_requests()
        self.assertEqual(gzip_call['params'][text_fixture('77616c6c6574')], 'account.unit')
        job = S3Job('abc', bytes(76), 1, 3, 10)
        request = dialect.submission_request(job, 'eA==', 12)
        self.assertEqual(request['method'], 'work.complete')
        self.assertEqual(s4_request(request)['method'], 'mining.submit')
        self.assertEqual(request['params'], {
            text_fixture('776f726b6572'): 'account.unit',
            'job_id': 'abc', text_fixture('706c61696e5f70726f6f66'): 'eA==',
        })

    def test_numeric_submission_and_device_report(self):
        first, second = bytes(range(8)), bytes(range(8, 16))
        submission = WRSubmission(first, second, 256, 256, 2048, 128, 8, 8, (1,), (2,)).request(9)
        self.assertEqual(submission['method'], text_fixture('7375626d6974'))
        fields, = submission['params']
        self.assertEqual(fields[text_fixture('6d696e65725f6e6f6e6365')], first.hex())
        self.assertEqual(fields[text_fixture('6e6f6e6365')], first.hex())
        self.assertEqual(fields[text_fixture('776f726b65725f6e6f6e6365')], second.hex())
        report = WRDialect.report_request('device', 25, 3)
        self.assertEqual(report['params'], [{text_fixture('6861736872617465'): 25, 'name': 'device'}])

    def test_cert_domain_and_legacy_protocol(self):
        self.assertEqual(s6_noise_seed_prefix(), bytes.fromhex('706561726c2f636572742d76332f6e6f6973652d736565642f'))
        self.assertEqual(s7_legacy_alias(), text_fixture('706561726c68617368'))

    def test_display_units_and_status(self):
        from workrunner.S3_training import S3_status, S3TrainingStats, format_work_rate
        from workrunner.protocol import TrainingConfig

        for index, unit in enumerate(('482f73', '6b482f73', '4d482f73', '47482f73',
                                      '54482f73', '50482f73', '45482f73')):
            self.assertEqual(format_work_rate(1000 ** index), '1.00 ' + text_fixture(unit))
        status = S3_status(S3TrainingStats(batches=1, attempts=2), TrainingConfig(), 1, None)
        self.assertEqual(status, text_fixture('48617368726174652861766729') + ' 134.22 ' +
                         text_fixture('4d482f73') + text_fixture(
                             '207c20475055204e2f41207c2061636365707465643d302072656a65637465643d30207c20'
                             '7375626d69747465643d3020626174636865733d3120617474656d7074733d32207374616c653d30'
                         ))
