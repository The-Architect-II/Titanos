"""Read public HTTPS model-page claims; never acquire models or execute page code.

Transport pins each connection to an address from the validated DNS answer while
TLS still authenticates the requested hostname. Redirects get fresh validation.
"""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import http.client
import ipaddress
import math
import queue
import re
import socket
import ssl
import threading
import time
from urllib.parse import parse_qsl, quote, urljoin, urlsplit, urlunsplit

_resolver_slots = threading.BoundedSemaphore(4)


def _public_address(value):
    address = ipaddress.ip_address(value)
    if (not address.is_global or address.is_multicast or address.is_reserved
            or address.is_loopback or address.is_link_local or address.is_unspecified):
        raise ValueError("Model pages must resolve only to public unicast addresses")
    if address.version == 6:
        if address.ipv4_mapped:
            _public_address(str(address.ipv4_mapped))
        if (address.sixtofour or address.teredo
                or address in ipaddress.ip_network("64:ff9b::/96")
                or address in ipaddress.ip_network("64:ff9b:1::/48")):
            raise ValueError("IPv6 transition addresses are not supported for model pages")
    return address


def _public_url(value):
    if not isinstance(value, str) or not value or re.search(r"[\x00-\x20\x7f\\]", value):
        raise ValueError("Model pages require a valid public HTTPS URL")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None:
        raise ValueError("Model pages require HTTPS without embedded credentials")
    for name, _ in parse_qsl(parts.query, keep_blank_values=True):
        name = re.sub(r"[^a-z0-9]", "", name.casefold())
        if (name in {"key", "apikey", "token", "accesstoken", "refreshtoken", "authtoken",
                     "secret", "clientsecret", "password", "passwd", "signature", "sig",
                     "authorization", "auth", "credential", "session", "sessionid", "sessiontoken"}
                or name.startswith(("xamz", "xgoog"))):
            raise ValueError("Use a public model-page URL without credential query parameters")
    host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    port = 443 if parts.port is None else parts.port
    if not 1 <= port <= 65535 or "%" in host:
        raise ValueError("Invalid model-page host or port")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if ("." not in host or host.endswith((".localhost", ".local", ".internal", ".home", ".lan"))
                or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in host.split("."))):
            raise ValueError("Model pages require a public hostname")
    else:
        _public_address(host)
    authority = f"[{host}]" if ":" in host else host
    if port != 443:
        authority += f":{port}"
    return urlunsplit(("https", authority, quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~"),
                       quote(parts.query, safe="%/:?@!$&'()*+,;=-._~"), ""))


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Model-page fetch exceeded its timeout")
    return remaining


def _resolve(host, port, deadline):
    # getaddrinfo has no portable timeout. A daemon preserves the caller's deadline.
    result = queue.Queue(maxsize=1)
    if not _resolver_slots.acquire(blocking=False):
        raise TimeoutError("Model-page DNS resolver capacity is busy")

    def resolve():
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception as exc:
            result.put(exc)
        finally:
            _resolver_slots.release()

    try:
        threading.Thread(target=resolve, daemon=True).start()
    except BaseException:
        _resolver_slots.release()
        raise
    try:
        addresses = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise TimeoutError("Model-page DNS resolution exceeded its timeout") from None
    if isinstance(addresses, Exception):
        raise ValueError("Model-page hostname could not be resolved") from None
    if not addresses:
        raise ValueError("Model-page hostname returned no addresses")
    for family, _, _, _, address in addresses:
        if family not in (socket.AF_INET, socket.AF_INET6):
            raise ValueError("Unsupported model-page address family")
        _public_address(address[0])
    return addresses


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, addresses, deadline):
        super().__init__(host, port, timeout=_remaining(deadline), context=ssl.create_default_context())
        self.addresses, self.deadline = addresses, deadline

    def connect(self):
        for family, kind, protocol, _, address in self.addresses:
            raw = socket.socket(family, kind, protocol)
            self.sock = raw
            try:
                raw.settimeout(_remaining(self.deadline))
                raw.connect(address)  # Numeric sockaddr: never resolve the hostname again.
            except OSError:
                raw.close()
                self.sock = None
                continue
            raw.settimeout(_remaining(self.deadline))
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
            return
        raise OSError("Could not connect to the public model-page host")


def _read_page(url, deadline, max_bytes):
    parts = urlsplit(url)
    addresses = _resolve(parts.hostname, parts.port or 443, deadline)
    connection = _PinnedHTTPS(parts.hostname, parts.port or 443, addresses, deadline)
    expired = threading.Event()

    def expire():
        expired.set()
        sock = connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    # Includes slow-drip HTTP headers/body; a per-read socket timeout alone does not.
    timer = threading.Timer(_remaining(deadline), expire)
    timer.daemon = True
    timer.start()
    try:
        target = urlunsplit(("", "", parts.path, parts.query, ""))
        connection.request("GET", target, headers={"User-Agent": "Titanos model-page importer",
                           "Accept": "text/html, application/xhtml+xml", "Accept-Encoding": "identity"})
        response = connection.getresponse()
        if response.status in (301, 302, 303, 307, 308):
            return response.status, dict(response.getheaders()), b""
        if response.status != 200:
            raise ValueError(f"Model-page server returned HTTP {response.status}")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Compressed model-page responses are not supported")
        content_type = response.getheader("Content-Type", "").split(";", 1)[0].lower().strip()
        if content_type not in ("text/html", "application/xhtml+xml"):
            raise ValueError("Model-page URL did not return an HTML page")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdigit() or int(length) > max_bytes):
            raise ValueError("Model-page response exceeds max_bytes or has invalid length")
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("Model-page response exceeds max_bytes")
        if length is not None and len(body) != int(length):
            raise ValueError("Model-page response ended before its declared length")
        _remaining(deadline)
        return response.status, dict(response.getheaders()), body
    except (OSError, http.client.HTTPException):
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Model-page fetch exceeded its timeout") from None
        raise ValueError("Model-page HTTPS request failed; check connectivity and TLS") from None
    finally:
        timer.cancel()
        connection.close()


class _PageMetadata(HTMLParser):
    def __init__(self, url, max_links, max_text_chars):
        super().__init__(convert_charrefs=True)
        self.url, self.max_links, self.max_text_chars = url, max_links, max_text_chars
        self.title, self.description = "", ""
        self.meta_title, self.license_claim = "", ""
        self.license_links, self.model_links = [], []
        self.in_title, self.skip, self.anchor = False, None, None
        self.seen = set()

    def clean(self, text):
        return " ".join(text.split())[:self.max_text_chars]

    def link(self, href, text="", rel=""):
        if not href or len(self.seen) >= self.max_links:
            return
        try:
            url = _public_url(urljoin(self.url, href))
        except (ValueError, UnicodeError):
            return
        descriptor = (urlsplit(url).path + " " + text + " " + rel).lower()
        license_link = "license" in descriptor or "licence" in descriptor or "copying" in descriptor
        host = urlsplit(url).hostname
        provider = next((name for domain, name in (("huggingface.co", "huggingface"),
                        ("github.com", "github"), ("modelscope.cn", "modelscope"),
                        ("kaggle.com", "kaggle"), ("ollama.com", "ollama"))
                        if host == domain or host.endswith("." + domain)), None)
        if not license_link and not provider and not re.search(r"model|checkpoint|weights|download|\.gguf|\.onnx|\.safetensors", descriptor):
            return
        key = (url, license_link)
        if key in self.seen:
            return
        self.seen.add(key)
        row = {"url": url, "label": self.clean(text), "provider_hint": provider,
               "evidence": "page-claim", "compatibility": "unverified"}
        (self.license_links if license_link else self.model_links).append(row)

    def handle_starttag(self, tag, attrs):
        if self.skip:
            return
        if tag in ("script", "style", "template"):
            self.skip = tag
            return
        values = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            name = (values.get("name") or values.get("property") or "").lower()
            content = self.clean(values.get("content") or "")
            if name in ("description", "og:description", "twitter:description") and not self.description:
                self.description = content
            elif name in ("og:title", "twitter:title") and not self.meta_title:
                self.meta_title = content
            elif name in ("license", "dc.rights", "dcterms.license"):
                self.license_claim = content
                if content.startswith("https://"):
                    self.link(content, "License", "license")
        elif tag == "a":
            self.anchor = [values.get("href"), "", values.get("rel") or ""]
        elif tag == "link":
            self.link(values.get("href"), values.get("title") or "", values.get("rel") or "")

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title:
            self.title = (self.title + data)[:self.max_text_chars]
        if self.anchor:
            self.anchor[1] = (self.anchor[1] + data)[:self.max_text_chars]

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip:
                self.skip = None
            return
        if tag == "title":
            self.in_title = False
        elif tag == "a" and self.anchor:
            self.link(*self.anchor)
            self.anchor = None


def fetch_model_page(url, *, timeout=15.0, max_bytes=2_000_000, max_redirects=3,
                     max_links=50, max_text_chars=2_000):
    """Return JSON-serializable page claims; never qualify, acquire or run a model.

    timeout is the whole fetch budget, including DNS, TLS, redirects and body.
    Proxies, cookies, auth, scripts and linked assets are never used or fetched.
    Errors do not echo URLs, which may contain caller-supplied query parameters.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    for name, value, minimum in (("max_bytes", max_bytes, 1), ("max_redirects", max_redirects, 0),
                                 ("max_links", max_links, 1), ("max_text_chars", max_text_chars, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    source = current = _public_url(url)
    deadline = time.monotonic() + timeout
    redirects = []
    for hop in range(max_redirects + 1):
        status, headers, body = _read_page(current, deadline, max_bytes)
        headers = {name.lower(): value for name, value in headers.items()}
        if status == 200:
            break
        if hop == max_redirects:
            raise ValueError("Model-page redirect limit exceeded")
        if not headers.get("location"):
            raise ValueError("Model-page redirect has no destination")
        current = _public_url(urljoin(current, headers["location"]))
        redirects.append(current)
    content_type = headers.get("content-type", "")
    charset = re.search(r"charset\s*=\s*[\"']?([a-zA-Z0-9._-]+)", content_type, re.I)
    try:
        html = body.decode(charset.group(1) if charset else "utf-8", "replace")
    except LookupError:
        html = body.decode("utf-8", "replace")
    page = _PageMetadata(current, max_links, max_text_chars)
    page.feed(html)
    page.close()
    return {"source_url": source, "final_url": current, "redirects": redirects,
            "content_sha256": hashlib.sha256(body).hexdigest(), "content_bytes": len(body),
            "content_type": content_type, "title": page.clean(page.title) or page.meta_title,
            "description": page.description, "license_claim": page.license_claim or None,
            "license_links": page.license_links, "model_links": page.model_links,
            "compatibility": "metadata-only", "qualification": None,
            "note": "Untrusted page claims; linked providers must verify license, exact revision, files and runtime support before acquisition."}
