# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image we labeled every commit in: its Python and SQLite
# decide which sqlglot tests run, so keep it if your image runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS map-builder
COPY requirements.lock /build/requirements.lock
RUN pip install --no-cache-dir -r /build/requirements.lock coverage==7.13.1
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz', '/tmp/sqlglot.tar.gz')" \
    && mkdir /src /coverage \
    && tar -xzf /tmp/sqlglot.tar.gz --strip-components=1 -C /src \
    && rm /tmp/sqlglot.tar.gz
COPY impact_collector.py /build/impact_collector.py
WORKDIR /src
ENV PYTHONPATH=/src:/build SKIP_INTEGRATION=0
RUN python -m pytest -n 4 --dist=loadfile -q -p impact_collector tests \
    && python /build/impact_collector.py
COPY statement_map.py /build/statement_map.py
RUN python /build/statement_map.py
COPY dependency_map.py /build/dependency_map.py
RUN python /build/dependency_map.py build

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90
COPY selector.py /app/selector.py
COPY commentry.py /app/commentry.py
COPY dependency_map.py /app/dependency_map.py
COPY canary.py /app/canary.py
COPY live_probe.py /app/live_probe.py
COPY test_categoriser.py /app/test_categoriser.py
COPY --from=map-builder /coverage/impact-map.json /app/impact-map.json
COPY --from=map-builder /coverage/dependency-map.json /app/dependency-map.json
COPY --from=map-builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=map-builder /usr/local/bin /usr/local/bin
COPY --from=map-builder /src/sqlglot /source/sqlglot
COPY --from=map-builder /src/tests /source/tests
RUN python /app/canary.py --build-dominance \
    --impact-map /app/impact-map.json \
    --source-root /source \
    --output /app/dominance-map.json
WORKDIR /app
CMD ["python", "/app/selector.py"]
