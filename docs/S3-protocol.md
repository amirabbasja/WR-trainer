# S3 session protocol

The KEX endpoint requires an **S4-compatible wire protocol** using
newline-terminated JSON objects over TCP or verified TLS. Neutral method
names describe application operations inside the trainer. `S3_methods.py`
contains the explicit compatibility mapping used by `S3Session` at request
serialization and notification dispatch:

| Application operation | Required wire method | Meaning |
|---|---|---|
| `compute.register` | `mining.subscribe` | Announce the client agent. |
| `compute.session` | `mining.authorize` | Authenticate the account and device; optionally request gzip. |
| `work.complete` | `mining.submit` | Send the completed work's native proof. |
| `work.update` | `mining.notify` | Receive a replacement job. |
| `compute.reconnect` | `client.reconnect` or `mining.reconnect` | End the session; reconnect only to the configured endpoint. |

All method names are complete, readable literals. No method-name encoding
or runtime prefix/suffix assembly is used. Neutral operation names are not
sent to the endpoint or accepted as incoming wire aliases. This mapping
does not change the separate WR numeric protocol.

The default handshake sends `mining.subscribe` with `[agent]`, immediately
followed by `mining.authorize` with `["account.device", password]`. The
first request may remain unanswered; it must not delay authentication.
Each request carries `jsonrpc: "2.0"` and a positive integer `id`.

An explicitly requested gzip session instead sends one `mining.authorize`
request with object parameters:

```json
{"wallet":"account.device","agent":"client/1","password":"x","type":"v2"}
```

Only a reply matching the authentication request ID, with literal
`result: true`, no error, and `type: "v2"`, enables gzip. Otherwise proof
bytes remain uncompressed. A completed proof uses `mining.submit` with
`{"worker":"account.device","job_id":"...","plain_proof":"BASE64"}`.
When negotiated, gzip compresses the decoded native proof bytes before
base64 encoding. Parameter names and proof serialization remain compatible
with the endpoint.

Incoming `mining.notify.params` is an object containing `job_id`, `header`,
`height`, `target`, and `cert_version`. The header contains 76 bytes encoded
as 152 hexadecimal characters. The target is a big-endian hexadecimal base
target. Missing certificate versions require an explicit configured
fallback; unknown versions fail. Replacement jobs invalidate previous work.

Responses retain the original request ID; only a literal successful result
with no error confirms completed work. There is no additional acknowledgement
RPC. Unknown extension notifications are ignored. Both reconnect methods
close the current session and invalidate its work; server-supplied redirect
addresses are never followed.
