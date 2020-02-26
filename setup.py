# _*_ coding: utf-8 _*_

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="bpa",
    version="0.2.0",
    author="Genpeng Xu",
    author_email="xgp1227@gmail.com",
    description="Building Project Assistant (BPA): An efficient & robust helper about building project, "
                "such as bill classification, bill similarity search, etc.",
    long_description=long_description,
    url="https://github.com/Genpeng/building-project-assistant",
    packages=setuptools.find_packages(),
    package_data={
        "bpa": [
            "db/standard-bills.txt",
            "dicts/stopwords.txt",
            "dicts/userdict.txt",
            "t1_model/label_2_type.dict",
            "t1_model/model.joblib",
            "t1_model/type_2_label.dict",
            "t1_model/vectorizer.joblib",
            "t2_model/database_vectors.joblib",
            "t2_model/ordinal_2_id.dict",
            "t2_model/vectorizer.joblib"
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        # "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
