
import setuptools
import os
import os


requires = """torch>=1.8.0
transformers>=4.10.0
datasets>=1.17.0
sentencepiece>=0.1.96
tqdm>=4.62.2
decorator
rich
web.py
gitpython
scipy # need?
scikit-learn # need?
delta_center_client==0.0.4
bigmodelvis
"""

def get_requirements():
    ret = [x for x in requires.split("\n") if len(x)>0]
    print("requirements:", ret)
    return ret



# path = os.path.dirname(os.path.abspath(__file__))
# requires =  get_requirements(path)

with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'opendelta',
        version = "0.3.2",
        description = "An open source framework for delta learning (parameter efficient learning).",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author = '',
        author_email = 'shengdinghu@gmail.com',
        license="Apache",
        url="https://github.com/thunlp/OpenDelta",
        keywords = ['PLM', 'Parameter-efficient-Learning', 'AI', 'NLP'],
        python_requires=">=3.6.0",
        install_requires=get_requirements(),
        package_dir={'opendelta':'opendelta'},
        package_data= {
            'opendelta':["utils/interactive/templates/*.html", 'requirments.txt'],
        },
        include_package_data=True,
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ]
    )
