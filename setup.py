from setuptools import find_packages, setup


setup(
    name="agfc",
    version="0.1.0",
    description="Standalone figure extraction and MinerU artifact repair with stable public contracts.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={"agfc": ["schemas/*.json"]},
    install_requires=["Pillow>=9", "PyMuPDF>=1.20"],
    entry_points={"console_scripts": ["agfc=agfc.cli:main"]},
    python_requires=">=3.9",
)
