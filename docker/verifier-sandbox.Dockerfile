FROM python:3.11.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

RUN useradd --uid 65534 --no-create-home --shell /usr/sbin/nologin sandbox || true
USER 65534:65534
WORKDIR /tmp
