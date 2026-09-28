# LLM Performance Benchmark CLI

A Python command-line tool for benchmarking and comparing Hugging Face causal language models across inference performance, system resource usage, and baseline generation-quality metrics.

## Planned capabilities

- Load three or more Hugging Face models from a YAML or JSON configuration file.
- Load benchmark prompts from CSV or JSONL datasets.
- Measure end-to-end inference latency and generation throughput.
- Monitor process RAM and CUDA VRAM when a compatible GPU is available.
- Calculate baseline quality metrics, including generated-token length, type-token ratio, and degenerate-output rate.
- Persist prompt-level results to CSV.
- Generate latency and memory comparison charts.
- Run consistently through Docker and Docker Compose.

## Project status

Implementation is in progress.

## Planned project structure

```text
llm-performance-benchmark-cli/
├── configs/       # Benchmark configuration files
├── datasets/      # Prompt datasets
├── outputs/       # Generated CSV reports and charts
├── src/           # Application source code
├── tests/         # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

Created for an academic machine-learning benchmarking assignment.