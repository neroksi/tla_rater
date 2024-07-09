from setuptools import find_packages, setup


def do_setup():
    install_requires = [
        "protobuf==5.26.1",
        "scikit-learn==1.4.2",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "matplotlib==3.8.4",
        "torch==2.3.0",
        "sentencepiece==0.2.0",
        "transformers==4.32.1",
        "tokenizers==0.13.2",
        "datasets==2.12.0",
        "accelerate==0.26.1",
        "ensemble-boxes==1.0.8",
        "PyYAML==6.0.1",
    ]

    extras = {}

    extras["test"] = []

    extras["all"] = sorted(
        set([rqrmt for _, flavour_rqrmts in extras.items() for rqrmt in flavour_rqrmts])
    )

    extras["dev"] = extras["all"]

    setup(
        name="rater",
        version="0.0.1",
        description="The Learning Agency RATER Competition.",
        author="Kossi N.",
        author_email="nkossy.pro@gmail.com",
        # url="#",
        packages=find_packages("src"),
        package_dir={"": "src"},
        # python_requires="==3.10.12",
        install_requires=install_requires,
        extras_require=extras,
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords="NER students essay education academy",
    )


if __name__ == "__main__":
    do_setup()
