from setuptools import setup, find_packages

## pip3 install torch torchvision torchaudio
## pip3 install -U "jax[cuda12]"

setup(
    name='weekend_learn',
    version='0.1',
    packages=find_packages(),
    description='Simple Playground for myself',
    author='Gyu Heo',
    author_email='heo9412@gmail.com',
    url='https://github.com/gyu-heo/weekend_learn',
    install_requires=[
        "optax",   # JAX optimization
        "flax",    # JAX neural networks
        "equinox",  # Equinox for JAX
        "scikit-learn",
        "pandas",
        "ipykernel",
        "babel",
        "black",
        "pytest",
        "datasets",
        "einops",
        "gymnasium",
        "jupyter",
        "matplotlib",
        "minigrid",
        "opt_einsum",
        "plotly",
        "swig",
        "wandb",
        "tensorboard",
        "tqdm"
    ],
    python_requires='>=3.9',
)
