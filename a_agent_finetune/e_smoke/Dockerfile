FROM external-agents-eval/pi-runtime:v0.84.2-arm64 AS pi
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git ripgrep fd-find libstdc++6 \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd \
    && rm -rf /var/lib/apt/lists/*
COPY --from=pi /opt/pi-runtime /opt/pi-runtime
ENV PATH="/opt/pi-runtime/bin:${PATH}"
ENTRYPOINT ["/opt/pi-runtime/bin/node", "/opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"]
