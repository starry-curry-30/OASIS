# OASIS

**Online Anchor-centric Scene Inference System for Open-Vocabulary Retrieval and Navigation in Dynamic Environments**

OASIS organizes multimodal scene memory around semantic anchors for online mapping, open-vocabulary retrieval, and navigation in dynamic environments.

## Source layout

- `oasis/`: core OASIS implementation.
- `applications/`: dataset, ROS, recording, and offline-query entry points.
- `config/`: system, dataset, navigation, and class-list configuration.
- `scripts/`: retrieval, navigation, evaluation, and analysis scripts.
- `utils/`: mapping, detection, memory management, navigation, and visualization utilities.

## Current snapshot

This repository contains the existing research source code. Datasets, model weights, credentials, Python caches, and generated experiment outputs are not included.

Before running experiments, configure local dataset/model paths and install the dependencies required by the selected entry point. Some entry points still import the external `dualmap` package. A complete standalone installation guide and end-to-end reproduction have not yet been validated for this snapshot.

Supply API credentials through environment variables, such as `GEMINI_API_KEY` or `GOOGLE_API_KEY`; do not commit credentials.
