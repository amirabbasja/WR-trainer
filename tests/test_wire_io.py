"""Native wire boundary tests, including memory ownership and real sockets."""

import ctypes
import gc
import io
import json
from pathlib import Path
import socket
import threading
import unittest

from workrunner.S6_codec import encode_frame
from workrunner.wire_io import (
    NativeChannel, WireError, WireLimitError, _Buffer, _library,
    _neutral_object, _transform, dumps_buffer, loads, read_json_body,
)


class ReadIntoOnly:
    def __init__(self, data, chunk=7, error_after=None):
        self.source = io.BytesIO(data)
        self.chunk = chunk
        self.calls = 0
        self.error_after = error_after
        self.retained = []

    def readinto(self, view):
        self.calls += 1
        if self.error_after == self.calls:
            self.retained.append(view[:])
            raise OSError('controlled read failure')
        count = min(len(view), self.chunk)
        return self.source.readinto(view[:count])


class NativeWireTests(unittest.TestCase):
    def test_rpc_translation_is_structural_and_preserves_values(self):
        raw = b'{"previousblockhash":"previousblockhash","transactions":[{"txid":"wallet","hash":"hash"}],"method":"getblocktemplate"}'
        self.assertEqual(loads(raw), {
            'parent_digest': 'previousblockhash',
            'records': [{'record_id': 'wallet', 'hash': 'hash'}],
            'method': 'template_request',
        })
        # Input belongs to the explicit compatibility caller and is untouched.
        self.assertIn(b'previousblockhash', raw)

    def test_protocol_translation_is_context_specific(self):
        value = loads(b'{"method":"mining.notify","params":["mining.notify",{"wallet":"hashrate","hash":"abc"}],"result":["extranonce","extranonce"]}', 's3')
        self.assertEqual(value['method'], 'training.notify')
        self.assertEqual(value['params'], ['mining.notify', {'account': 'hashrate', 'hash': 'abc'}])
        self.assertEqual(value['result'], ['extranonce', 'extranonce'])
        self.assertEqual(loads(b'{"result":["extranonce","extranonce"]}', 'wr'),
                         {'result': ['extracounter', 'extranonce']})
        self.assertEqual(loads(b'{"method":"unlisted","wallet":"abc"}', 'wr'),
                         {'method': 'unlisted', 'account': 'abc'})

    def test_escaped_keys_utf8_and_surrogate_pairs(self):
        self.assertEqual(loads(b'{"previousblockha\\u0073h":"\\ud83d\\ude00","\xc3\xa9":"\\u0000"}'),
                         {'parent_digest': '\U0001f600', '\u00e9': '\x00'})
        with dumps_buffer({'parent_digest': '\U0001f600', 'x': 'a\nb\\c"'}) as view:
            self.assertEqual(json.loads(view.tobytes()), {'previousblockhash': '\U0001f600', 'x': 'a\nb\\c"'})

    def test_duplicate_and_translated_collision_keys_rejected(self):
        for raw in [b'{"x":1,"x":2}', b'{"x":1,"\\u0078":2}',
                    b'{"previousblockhash":"a","parent_digest":"b"}',
                    b'{"outer":{"txid":1,"record_id":2}}']:
            with self.subTest(raw=raw), self.assertRaises(WireError):
                loads(raw)

    def test_invalid_json_numbers_utf8_and_depth_rejected_without_content(self):
        values = [b'{"secret":NaN}', b'{"secret":Infinity}', b'{"secret":1e999}',
                  b'{"secret":01}', b'{"secret":1.}', b'{"secret":.1}',
                  b'{"secret":true} trailing', b'{"secret":"\xc0\xaf"}',
                  b'{"secret":"\xed\xa0\x80"}', b'{"secret":"\\ud800"}',
                  b'{"secret":"\\udc00"}', b'{"secret":"\\ud800\\u0041"}',
                  b'{"secret":"line\nbreak"}', b'{"secret":}', b'\xef\xbb\xbf{}']
        for value in values:
            with self.subTest(value=value), self.assertRaises(WireError) as caught:
                loads(value)
            self.assertNotIn('secret', str(caught.exception))
        with self.assertRaises(WireLimitError):
            loads(b'[' * 130 + b'0' + b']' * 130)

    def test_json_numbers_and_empty_containers(self):
        value = b'{"a":-12,"b":0,"c":1.25e-3,"d":123456789012345678901234567890,"e":{},"f":[],"g":null,"h":false}'
        self.assertEqual(loads(value), json.loads(value))
        self.assertEqual(loads(b'[1e-400,-1e-400,1e-324]'), json.loads(b'[1e-400,-1e-400,1e-324]'))

    def test_rpc_serialization_never_translates_arbitrary_values(self):
        with dumps_buffer({'method': 'tip_request', 'parent_digest': 'parent_digest', 'params': ['template_request']}) as view:
            self.assertEqual(json.loads(view.tobytes()), {
                'method': 'getbestblockhash', 'previousblockhash': 'parent_digest', 'params': ['template_request'],
            })

    def test_wr_seeded_frame_matches_existing_codec(self):
        neutral = {'id': 1, 'method': 'report_info', 'params': [{'prochPerSecond': 123.5, 'account': 'demo'}]}
        expected = {'id': 1, 'method': 'report_info', 'params': [{'hashrate': 123.5, 'wallet': 'demo'}]}
        with dumps_buffer(neutral, 'wr', framed=True, seed=0x10203040) as view:
            self.assertEqual(view.tobytes(), encode_frame(expected, seed=0x10203040))

    def test_all_captured_wr_frames_decode_to_neutral_messages(self):
        fixture = Path(__file__).with_name('fixtures') / 'wr_S6_codec_frames.json'
        if not fixture.is_file():
            self.skipTest('captured frame fixture is kept in the author repository')
        for entry in json.loads(fixture.read_text()):
            raw = entry['wire'].encode()
            transformed = _transform(ctypes.c_char_p(raw), len(raw), 'wr', framed=True)
            try:
                expected = json.loads(json.dumps(entry['message']))
                if isinstance(expected.get('result'), list) and expected['result'][0] == 'extranonce':
                    expected['result'][0] = 'extracounter'
                for params in expected.get('params', []):
                    if isinstance(params, dict) and 'hashrate' in params:
                        params['prochPerSecond'] = params.pop('hashrate')
                self.assertEqual(_neutral_object(transformed), expected)
            finally:
                transformed.close()

    def test_malformed_wr_frame_rejects_and_frees_native_scratch(self):
        baseline = _library().wr_wire_live_buffers()
        for raw in [b'zzzzzzzz12', b'00000000', b'000000000', b'00000000ff', b'not hex']:
            with self.subTest(raw=raw), self.assertRaises(WireError):
                _transform(ctypes.c_char_p(raw), len(raw), 'wr', framed=True)
        gc.collect()
        self.assertEqual(_library().wr_wire_live_buffers(), baseline)

    def test_scoped_wire_views_are_wiped_and_retained_derived_views_are_safe(self):
        library = _library()
        baseline = library.wr_wire_live_buffers()
        with dumps_buffer({'parent_digest': 'private-value'}) as primary:
            derived = primary[:]
            self.assertIn(b'previousblockhash', bytes(derived))
        self.assertEqual(bytes(derived), bytes(len(derived)))
        with self.assertRaises(ValueError):
            len(primary)
        self.assertEqual(library.wr_wire_live_buffers(), baseline + 1)
        derived.release()
        gc.collect()
        self.assertEqual(library.wr_wire_live_buffers(), baseline)

    def test_exception_wipes_retained_output_and_frees_after_final_release(self):
        baseline = _library().wr_wire_live_buffers()
        with self.assertRaisesRegex(OSError, 'controlled'):
            with dumps_buffer({'account': 'private'}, 's3') as view:
                retained = view[:]
                raise OSError('controlled send failure')
        self.assertEqual(bytes(retained), bytes(len(retained)))
        retained.release()
        gc.collect()
        self.assertEqual(_library().wr_wire_live_buffers(), baseline)

    def test_native_buffer_growth_clear_and_borrowed_view_guard(self):
        library = _library()
        before = library.wr_wire_wipe_calls()
        buffer = _Buffer()
        try:
            buffer.append_pointer(ctypes.c_char_p(b'secret'), 6)
            buffer.reserve(4096)
            self.assertGreater(library.wr_wire_wipe_calls(), before)
            with buffer.view() as view:
                retained = view[:]
            with self.assertRaisesRegex(RuntimeError, 'borrowed view'):
                buffer.reserve(8192)
            buffer.clear()
            self.assertEqual(bytes(retained), bytes(6))
            retained.release()
            buffer.reserve(8192)
        finally:
            buffer.close()

    def test_http_readinto_only_fragmentation_and_bounded_read(self):
        self.assertEqual(read_json_body(ReadIntoOnly(b'{"previousblockhash":"abc"}'), 100),
                         {'parent_digest': 'abc'})
        with self.assertRaises(WireLimitError):
            read_json_body(ReadIntoOnly(b' ' * 101), 100)

    def test_http_read_failure_wipes_retained_native_input(self):
        source = ReadIntoOnly(b'{"previousblockhash":"abc"}', error_after=2)
        baseline = _library().wr_wire_live_buffers()
        with self.assertRaises(OSError):
            read_json_body(source, 100)
        self.assertTrue(source.retained)
        self.assertEqual(bytes(source.retained[0]), bytes(len(source.retained[0])))
        source.retained[0].release()
        source.retained.clear()
        gc.collect()
        self.assertEqual(_library().wr_wire_live_buffers(), baseline)

    def test_socket_pair_fragmented_reads_multiple_frames_and_single_write(self):
        client, peer = socket.socketpair()
        with client, peer:
            client.settimeout(2)
            peer.settimeout(2)
            channel = NativeChannel(client, 's3')
            try:
                peer.sendall(b'{"method":"mini')
                peer.sendall(b'ng.notify","params":[]}\n{ "id": 2, "result":true}\n\n')
                self.assertEqual(channel.receive(), {'method': 'training.notify', 'params': []})
                self.assertEqual(channel.receive(), {'id': 2, 'result': True})
                self.assertIsNone(channel.receive())
                channel.send({'method': 'training.submit', 'params': [{'account': 'abc'}]})
                self.assertEqual(json.loads(peer.recv(4096)),
                                 {'method': 'mining.submit', 'params': [{'wallet': 'abc'}]})
            finally:
                channel.close()

    def test_wr_diagnostics_are_encoded_and_explicitly_discarded(self):
        client, peer = socket.socketpair()
        with client, peer:
            channel = NativeChannel(client, 'wr', max_frame=100)
            try:
                peer.sendall(b'bad-frame\n')
                with self.assertRaises(WireError):
                    channel.receive()
                self.assertEqual(channel.last_frame(), b'bad-frame\n')
                channel.discard_frame()
                self.assertEqual(channel.last_frame(), b'')
            finally:
                channel.close()

    def test_close_during_receive_defers_free_without_blocking_or_dangling_view(self):
        started, release = threading.Event(), threading.Event()
        retained, failures = [], []

        class Connection:
            def recv_into(self, view):
                retained.append(view[:])
                started.set()
                release.wait(2)
                return 0

        channel = NativeChannel(Connection(), 's3')
        baseline = _library().wr_wire_live_buffers()

        def receive():
            try:
                channel.receive()
            except BaseException as error:
                failures.append(type(error))

        thread = threading.Thread(target=receive)
        thread.start()
        self.assertTrue(started.wait(2))
        channel.close()
        self.assertEqual(_library().wr_wire_live_buffers(), baseline)
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [ConnectionError])
        self.assertEqual(bytes(retained[0]), bytes(len(retained[0])))
        retained[0].release()
        retained.clear()
        gc.collect()
        self.assertEqual(_library().wr_wire_live_buffers(), baseline - 3)


if __name__ == '__main__':
    unittest.main()
