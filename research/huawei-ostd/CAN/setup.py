from setuptools import find_packages, setup

exec(open("can/version.py").read())


def read_requirements(fps):
    reqs = []
    for fp in fps:
        with open(fp) as f:
            reqs.extend(f.readlines())

    return reqs


setup(
    name="can",
    description="A toolbox of CAN models and algorithms based on MindSpore.",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["can", "can.*"]),
    install_requires=read_requirements(["requirements.txt"]),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    tests_require=[
        "pytest",
    ],
    version=__version__,
    zip_safe=False,
)
