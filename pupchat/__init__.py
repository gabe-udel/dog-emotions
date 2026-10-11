"""Pupchat: video + question -> keypoints -> features -> evidence -> OpenAI -> answer.

A layer on top of the measurement pipeline in src/. It changes no model, training or
evaluation code; it reads what that pipeline produces and describes it.

Prototype only: DogFLW is CC BY-NC 4.0 and the SuperAnimal weights follow the
DeepLabCut Model Zoo terms, so this must not be deployed publicly.
"""
__all__ = ["config", "reliability", "evidence", "pipeline", "client"]
