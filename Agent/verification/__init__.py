"""Agent verification layer.

Additive, read-only guardrails: gates flag and report, they never modify
pipeline outputs (failure -> more human review, never more automation).
"""
