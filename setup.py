from setuptools import setup, find_packages

setup(
    name="mocha-tattoo-removal-automation",
    version="0.1.0",
    description="Automation tools to split tracking blocks and export Mocha Pro templates for tattoo removal workflows.",
    author="MarcosRRJ",
    license="MIT",
    py_modules=[
        "mocha_block_splitter",
        "occlusion_detector",
        "tracking_optimizer",
        "mocha_script_exporter",
        "mocha_config_generator",
        "example_workflow",
    ],
    install_requires=[
        "numpy",
        "opencv-python",
        "pandas",
        "tqdm",
        "scikit-image",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mocha-block-splitter=mocha_block_splitter:_cli",
        ]
    },
)