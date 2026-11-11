# Course code image "scipy": the pinned Jupyter Docker Stacks image plus the
# course's extra packages (requirements-scipy.txt). Build with: make code-images
# Run one file in it with: python3 tools/run_code.py code/<file>
FROM quay.io/jupyter/scipy-notebook:2026-11-02

# Graphviz's "dot" program, which the graphviz Python package calls to draw.
# Not version-pinned: Ubuntu's archive drops superseded versions, so a pin
# would break the build after the next security update.
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends graphviz \
 && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

# Install the extras against the base image's own packages: every package the
# base image ships (numpy, pandas, scikit-learn and the rest) is a constraint,
# so the build fails rather than replace one of them. The one exception is
# setuptools, which TPOT needs older (see requirements-scipy.txt).
COPY --chown=${NB_UID}:${NB_GID} requirements-scipy.txt /tmp/requirements.txt
RUN pip list --format=freeze | grep -v '^setuptools==' > /tmp/base-constraints.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/base-constraints.txt \
 && pip check \
 && rm /tmp/requirements.txt /tmp/base-constraints.txt
