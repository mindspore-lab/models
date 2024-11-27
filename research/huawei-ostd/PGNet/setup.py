from setuptools import find_packages, setup

def read_requirements(fps):
    reqs = []
    for fp in fps:
        with open(fp) as f:
            reqs.extend(f.readlines())

    return reqs


setup(
    name="models",
    author="MindSpore Ecosystem",
    author_email="mindspore-ecosystem@example.com",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["models", "models.*"]),
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
    zip_safe=False,
)
