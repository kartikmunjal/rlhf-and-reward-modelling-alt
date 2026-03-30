from setuptools import setup, find_packages

setup(
    name="rlhf-pipeline",
    version="0.1.0",
    author="Kartik Munjal",
    description="End-to-end RLHF pipeline: SFT + Reward Modeling + PPO vs DPO on GPT-2",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "trl>=0.7.4",
        "accelerate>=0.25.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff"],
        "tracking": ["wandb>=0.16.0"],
        "vis": ["matplotlib>=3.8.0", "seaborn>=0.13.0", "plotly>=5.18.0"],
    },
)
