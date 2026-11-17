"""Query-time package: retrieve chunks, build the prompt, stream the answer.

``search.py`` finds the chunks, ``prompt.py`` turns them into messages, and
``generate.py`` runs the two and streams the model's answer. Import from
those modules directly; this file only marks the folder as a package.
"""
