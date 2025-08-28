# Architecture Overview

- **Stage I**: 3D Swin U‑Net + Cross‑Phase Attention Fusion (dual encoders L1–L3; bottleneck; decoder; 1×1×1 head → P; threshold → M0).
- **Stage II**: MARL refinement with Expansion/Contraction agents, PPO, topology‑aware reward → final mask M*.
- **Stage III**: Grad‑CAM++ on last decoder features → heatmap overlays on CT; M* contour optional.
