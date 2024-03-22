import pathlib
import setuptools

setuptools.setup(
    name="llmFunctionDecorator",
    version="1.0.0",
    description="A simplified function decorator for OpenAI and LiteLLM API calls.",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/A-M-D-R-3-W/llmFunctionDecorator/",
    author="A-M-D-R-3-W",
    author_email="AMDR3W@proton.me",
    license="MIT",
    project_urls={
        "Documentation": "https://github.com/A-M-D-R-3-W/llmFunctionDecorator/blob/main/README.md",
        "Source": "https://github.com/A-M-D-R-3-W/llmFunctionDecorator/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    include_package_data=True,
)