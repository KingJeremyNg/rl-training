from setuptools import setup

setup(
    name="rl-training",
    version="1.0",
    description="Reinforcement Learning Tutorial",
    author="Jeremy Ng",
    packages=["rl-training"],
    package_dir={"rl-training": "src"},
    # external packages as dependencies
    install_requires=["rl-training", "swig",
                      "gymnasium", "stable-baselines3", "ale-py", "python-dotenv"],
)
