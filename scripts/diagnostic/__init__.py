"""Diagnostic and probe scripts. Read-only checks and one-off investigations.

Separate from scripts/, which holds the pipeline entrypoints that produce or
mutate project state: acquisition, extraction, embedding, evaluation runs.
Nothing here is part of a pipeline; each file answers a question about the
system and is safe to run at any time.
"""