# Source package

Place reusable implementation code here rather than duplicating it across
notebooks and scripts. Suggested modules as the project grows:

```text
src/data/          # Audio loading, validation, and manifests
src/models/        # MMS wrappers and experiment models
src/evaluation/    # Metrics and result aggregation
src/pipelines/     # End-to-end inference/evaluation workflows
```
