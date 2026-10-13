"""Human adjudication: labelling queues and scoring against human labels.

Downstream of classification (spec §2): reads finished classification results,
never imported by `classifier`. Article text from the corpus goes into local,
git-ignored queue files only.
"""
