"""Run reports: cost/time/throughput, label distribution, cross-model agreement,
and synthetic dev-set accuracy.

Downstream of the classifier: it reads finished classification results and never
writes into classification_results/. No module in `classifier` except the CLI
orchestration imports this package.
"""
