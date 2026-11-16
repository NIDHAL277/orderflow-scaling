# syntax=docker/dockerfile:1.7
# Caddy with the rate-limit module (not part of the standard build).
FROM caddy:2-builder AS builder
# Optional extra root certificate, for a build behind a TLS-inspecting proxy or antivirus
# (docs/operations.md): docker build --secret id=extra_ca,src=<root.pem> ...
# Without the secret this step only builds Caddy.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then \
      cat /run/secrets/extra_ca >> /etc/ssl/certs/ca-certificates.crt; \
    fi; \
    xcaddy build --with github.com/mholt/caddy-ratelimit

FROM caddy:2
COPY --from=builder /usr/bin/caddy /usr/bin/caddy
