"""Filtered inference egress for the A container (stdlib only; task images ship Python 3.5+).

Same policy as Pier's Squid egress proxy used by the historical Pi A runs: Basic
authentication as user ``agent`` with a per-task token, and only allowlisted host
names are reachable.  The API is HTTPS, so only ``CONNECT <allowlisted>:443`` is
accepted.  Every decision is logged without the token.
"""
import base64
import os
import select
import socket
import sys
import threading
import time

ALLOWED = set(
    name.strip().lower() for name in os.environ["ALLOWLIST_DOMAINS"].split(",") if name.strip()
)
PORT = int(os.environ.get("PROXY_PORT", "8080"))
EXPECTED = "Basic " + base64.b64encode(
    ("agent:" + os.environ["PROXY_TOKEN"]).encode("ascii")
).decode("ascii")
IDLE_TIMEOUT_SEC = 900  # Squid's default read_timeout


def log(*parts):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sys.stdout.write(" ".join([stamp] + [str(part) for part in parts]) + "\n")
    sys.stdout.flush()


def reply(connection, status, reason):
    extra = 'Proxy-Authenticate: Basic realm="PierPolicyProxy"\r\n' if status == 407 else ""
    message = "HTTP/1.1 {0} {1}\r\nContent-Length: 0\r\nConnection: close\r\n{2}\r\n".format(
        status, reason, extra)
    try:
        connection.sendall(message.encode("ascii"))
    except OSError:
        pass


def relay(client, upstream):
    sockets = [client, upstream]
    try:
        while True:
            readable, _, _ = select.select(sockets, [], [], IDLE_TIMEOUT_SEC)
            if not readable:
                return "idle_timeout"
            for source in readable:
                data = source.recv(65536)
                if not data:
                    return "closed"
                (upstream if source is client else client).sendall(data)
    except OSError:
        return "socket_error"


def handle(client, address):
    upstream = None
    try:
        client.settimeout(30)
        head = b""
        while b"\r\n\r\n" not in head and len(head) < 65536:
            chunk = client.recv(4096)
            if not chunk:
                break
            head += chunk
        header, _, rest = head.partition(b"\r\n\r\n")
        lines = header.decode("latin-1").split("\r\n")
        request = lines[0].split()
        if len(request) < 2:
            log("DENY", "malformed", address[0])
            reply(client, 400, "Bad Request")
            return
        method, target = request[0].upper(), request[1]
        headers = {}
        for line in lines[1:]:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
        if headers.get("proxy-authorization") != EXPECTED:
            log("DENY", "auth", method, target)
            reply(client, 407, "Proxy Authentication Required")
            return
        host, _, port = target.rpartition(":")
        host = host.strip("[]").lower()
        if method != "CONNECT" or host not in ALLOWED or port != "443":
            log("DENY", "policy", method, target)
            reply(client, 403, "Forbidden")
            return
        upstream = socket.create_connection((host, 443), timeout=30)
        upstream.settimeout(None)
        client.settimeout(None)
        client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if rest:
            upstream.sendall(rest)
        log("ALLOW", "CONNECT", target)
        log("CLOSE", target, relay(client, upstream))
    except Exception as error:  # noqa: BLE001 - one connection must not stop the proxy
        log("ERROR", type(error).__name__)
        reply(client, 502, "Bad Gateway")
    finally:
        for item in (client, upstream):
            if item is not None:
                try:
                    item.close()
                except OSError:
                    pass


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(64)
    log("READY", "port", PORT, "allow", ",".join(sorted(ALLOWED)))
    while True:
        client, address = server.accept()
        thread = threading.Thread(target=handle, args=(client, address))
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    main()
