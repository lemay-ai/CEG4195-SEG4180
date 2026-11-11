# Course code image "tensorflow": the pinned Jupyter Docker Stacks image plus the
# course's extra packages (requirements-tensorflow.txt). Build with: make code-images
# Run one file in it with: python3 tools/run_code.py code/<file>
FROM quay.io/jupyter/tensorflow-notebook:2026-11-02

# Install the extras against the base image's own packages: every package the
# base image ships (numpy, pandas, scikit-learn and the rest) is a constraint,
# so the build fails rather than replace one of them.
COPY --chown=${NB_UID}:${NB_GID} requirements-tensorflow.txt /tmp/requirements.txt
RUN pip list --format=freeze > /tmp/base-constraints.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/base-constraints.txt \
 && pip check \
 && rm /tmp/requirements.txt /tmp/base-constraints.txt
