"""Focused metadata and SSRF regression checks; no model/runtime dependencies."""
import hashlib
from io import BytesIO
from pathlib import Path
import socket
import ssl
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Runs from this isolated staged tree and from the normal source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from titanos import model_page as page


HTML = b'''<html><head><title>Small &amp; useful model</title>
<meta name="description" content="A model page, not proof of support.">
<meta name="license" content="apache-2.0">
<link rel="license" href="/LICENSE"></head><body>
<script>window.executed=true; '<a href="https://evil.example/model">Bad</a>';</script>
<a href="https://huggingface.co/org/model/tree/abc">Model checkpoint</a>
<a href="https://github.com/k2-fsa/sherpa-onnx">Sherpa provider</a>
<a href="/model.onnx">Download ONNX</a><a href="/about">About</a>
<a href="https://127.0.0.1/model">Private</a>
<a href="javascript:alert(1)">Download</a>
<a href="https://user:pass@public.example/model">Credential link</a>
</body></html>'''


def addresses(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port))]


class Response:
    status = 200

    def __init__(self, body=HTML, headers=None, status=200):
        self.status = status
        self.headers = {"Content-Type": "text/html; charset=utf-8", **(headers or {})}
        self.stream = BytesIO(body)

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def getheaders(self):
        return list(self.headers.items())

    def read(self, amount):
        return self.stream.read(amount)


class Checks(unittest.TestCase):
    def fetch(self, response, **kwargs):
        with patch.object(page.socket, "getaddrinfo", side_effect=addresses), patch.object(page, "_PinnedHTTPS") as connection:
            connection.return_value.getresponse.return_value = response
            return page.fetch_model_page("https://models.example/card#section", **kwargs)

    def test_metadata_preserves_source_hash_and_disclaims_compatibility(self):
        result = self.fetch(Response())
        self.assertEqual(result["source_url"], "https://models.example/card")
        self.assertEqual(result["content_sha256"], hashlib.sha256(HTML).hexdigest())
        self.assertEqual(result["title"], "Small & useful model")
        self.assertEqual(result["description"], "A model page, not proof of support.")
        self.assertEqual(result["license_claim"], "apache-2.0")
        self.assertEqual(result["license_links"][0]["url"], "https://models.example/LICENSE")
        self.assertEqual(len(result["model_links"]), 3)
        self.assertEqual(result["compatibility"], "metadata-only")
        self.assertIsNone(result["qualification"])
        self.assertTrue(all(row["compatibility"] == "unverified" for row in result["model_links"]))
        self.assertNotIn("evil.example", str(result))
        self.assertNotIn("window.executed", str(result))

    def test_bad_initial_urls_never_reach_transport(self):
        with patch.object(page, "_read_page") as transport:
            for url in ("http://example.com", "https://user:pw@example.com", "https://127.0.0.1/",
                        "https://169.254.169.254/", "https://[::1]/", "https://[::ffff:127.0.0.1]/",
                        "https://[64:ff9b::a00:1]/", "https://224.0.0.1/", "https://service.local/",
                        "https://example.com\\@127.0.0.1/", "https://example.com/\r\nX-Test:bad"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    page.fetch_model_page(url)
            transport.assert_not_called()

    def test_credential_queries_never_fetch_or_enter_metadata(self):
        with patch.object(page, "_read_page") as transport:
            for key in ("token", "api_key", "%61ccess_token", "X-Amz-Signature", "session_id"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "credential query"):
                    page.fetch_model_page("https://models.example/card?" + key + "=private-value")
            transport.assert_not_called()
        with patch.object(page, "_read_page", return_value=(302, {"Location": "/model?token=private-value"}, b"")) as transport:
            with self.assertRaisesRegex(ValueError, "credential query"):
                page.fetch_model_page("https://models.example/card")
            self.assertEqual(transport.call_count, 1)
        body = HTML.replace(b"</body>", b'<a href="/download?token=private-value">Model</a></body>')
        result = self.fetch(Response(body))
        self.assertNotIn("private-value", str(result))
        with patch.object(page, "_read_page", return_value=(200, {"Content-Type": "text/html"}, HTML)) as transport:
            result = page.fetch_model_page("https://models.example/card?model=small&revision=v1")
            self.assertEqual(result["source_url"], "https://models.example/card?model=small&revision=v1")
            self.assertEqual(transport.call_args.args[0], result["source_url"])

    def test_all_dns_answers_are_checked_before_connect(self):
        mixed = addresses(None, 443) + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]
        with patch.object(page.socket, "getaddrinfo", return_value=mixed), patch.object(page, "_PinnedHTTPS") as connection:
            with self.assertRaisesRegex(ValueError, "public"):
                page.fetch_model_page("https://models.example/card")
            connection.assert_not_called()

    def test_connection_pins_numeric_address_preserves_hostname_tls(self):
        context, raw = MagicMock(), MagicMock()
        context.check_hostname, context.verify_mode = True, ssl.CERT_REQUIRED
        with patch.object(page.ssl, "create_default_context", return_value=context) as tls, patch.object(page.socket, "socket", return_value=raw), patch.object(page.socket, "getaddrinfo", side_effect=AssertionError("No second DNS lookup")):
            connection = page._PinnedHTTPS("models.example", 443, addresses(None, 443), time.monotonic() + 5)
            connection.connect()
            tls.assert_called_once_with()
            raw.connect.assert_called_once_with(("93.184.216.34", 443))
            context.wrap_socket.assert_called_once_with(raw, server_hostname="models.example")
            self.assertTrue(context.check_hostname)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            connection.close()

    def test_redirects_revalidate_and_keep_provenance(self):
        with patch.object(page, "_read_page", side_effect=[(302, {"Location": "/final"}, b""), (200, {"Content-Type": "text/html"}, HTML)]):
            result = page.fetch_model_page("https://models.example/card")
            self.assertEqual(result["final_url"], "https://models.example/final")
            self.assertEqual(result["redirects"], [result["final_url"]])
        for target in ("https://10.1.2.3/model", "http://example.com/model", "https://user:pw@example.com"):
            with patch.object(page, "_read_page", return_value=(302, {"Location": target}, b"")) as transport:
                with self.assertRaises(ValueError):
                    page.fetch_model_page("https://models.example/card")
                self.assertEqual(transport.call_count, 1)
        with patch.object(page, "_read_page", return_value=(302, {"Location": "/again"}, b"")) as transport:
            with self.assertRaisesRegex(ValueError, "redirect limit"):
                page.fetch_model_page("https://models.example/card", max_redirects=1)
            self.assertEqual(transport.call_count, 2)

    def test_redirect_hostname_private_dns_is_blocked(self):
        answers = [addresses(None, 443), [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]]
        with patch.object(page.socket, "getaddrinfo", side_effect=answers), patch.object(page, "_PinnedHTTPS") as connection:
            connection.return_value.getresponse.return_value = Response(status=302, headers={"Location": "https://other.example/model"})
            with self.assertRaisesRegex(ValueError, "public"):
                page.fetch_model_page("https://models.example/card")
            self.assertEqual(connection.call_count, 1)

    def test_limits_and_non_html_are_rejected(self):
        for response in (Response(headers={"Content-Length": "99999"}), Response(headers={"Content-Encoding": "gzip"}),
                         Response(headers={"Content-Type": "application/octet-stream"}), Response(b"x" * 1001),
                         Response(b"short", headers={"Content-Length": "100"})):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.fetch(response, max_bytes=1000)
        result = self.fetch(Response(), max_links=1, max_text_chars=10)
        self.assertLessEqual(len(result["title"]), 10)
        self.assertEqual(len(result["license_links"]) + len(result["model_links"]), 1)

    def test_dns_and_slow_body_use_total_deadline(self):
        def slow_dns(*args, **kwargs):
            time.sleep(0.15)
            return addresses(None, 443)
        started = time.monotonic()
        with patch.object(page.socket, "getaddrinfo", side_effect=slow_dns), self.assertRaises(TimeoutError):
            page.fetch_model_page("https://models.example/card", timeout=0.02)
        self.assertLess(time.monotonic() - started, 0.12)
        class SlowResponse(Response):
            def read(self, amount):
                time.sleep(0.04)
                return b""
        with self.assertRaises(TimeoutError):
            self.fetch(SlowResponse(), timeout=0.02)

    def test_config_limits_are_validated(self):
        for options in ({"timeout": 0}, {"timeout": float("inf")}, {"max_bytes": 0}, {"max_redirects": -1},
                        {"max_links": True}, {"max_text_chars": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                page.fetch_model_page("https://models.example/card", **options)

    def test_repeated_dns_timeouts_keep_resolver_threads_bounded(self):
        release=threading.Event();baseline=threading.active_count()
        def stuck(*args,**kwargs):
            release.wait(2);return addresses(None,443)
        try:
            with patch.object(page.socket,"getaddrinfo",side_effect=stuck):
                for _ in range(8):
                    with self.assertRaises(TimeoutError):
                        page.fetch_model_page("https://models.example/card",timeout=.01)
                self.assertLessEqual(threading.active_count(),baseline+4)
        finally:
            release.set()
            for _ in range(100):
                if threading.active_count()<=baseline:break
                time.sleep(.01)


if __name__ == "__main__":
    unittest.main()
