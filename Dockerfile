FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir '.[models]'
# Non-root user. TITANOS_HOME (registry, pulled weights, installed runtimes) is a volume so it outlives the container;
# the directory is created with the user's ownership so a fresh named volume inherits it.
RUN useradd --create-home --uid 10001 titanos \
    && mkdir -p /var/lib/titanos && chown titanos:titanos /var/lib/titanos
USER titanos
ENV TITANOS_HOME=/var/lib/titanos
VOLUME ["/var/lib/titanos"]
EXPOSE 11435
# /healthz is answered without a key in both key modes; the slim image has no curl, so the interpreter probes it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); sys.exit(0 if o.open('http://127.0.0.1:'+os.environ.get('TITANOS_PORT','11435')+'/healthz', timeout=4).status==200 else 1)"]
ENTRYPOINT ["titanos"]
# serve reads the registry before it binds (run init once on a fresh volume), binds every container interface, and the CLI
# refuses that without TITANOS_API_KEY (docs/GETTING_STARTED.md, Docker):
#   docker run --rm -v titanos-home:/var/lib/titanos titanos init
#   docker run --rm -e TITANOS_API_KEY=change-me -p 127.0.0.1:11435:11435 -v titanos-home:/var/lib/titanos titanos
CMD ["serve", "--host", "0.0.0.0"]
