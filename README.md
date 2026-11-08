# SlangGuard

A continual-learning MLOps system for Indonesian sentiment analysis.
When the language in new data changes (drift), the system retrains the model
with LoRA + replay, so it learns new slang without forgetting old data.

## Status

Project setup is done. Data preparation is next.
See [docs/design.md](docs/design.md) for the full design and roadmap.
