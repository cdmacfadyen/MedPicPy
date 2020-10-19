import setuptools

from distutils.core import setup
"""
New release checklist:
    - update version
    - pull and push
    - run python3 setup.py sdist bdist_wheel
    - python3 -m twine upload dist/*
"""
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'medpicpy',         
    packages = ['medpicpy'],   
    version = '1.0.3',      
    license='MIT',       
    description = 'A package to simplify loading medical imaging datasets.',
    long_description=long_description,
    long_description_content_type="text/markdown",   
    author = 'Craig Macfadyen',                  
    author_email = 'cdmacfadyen@gmail.com',      
    url = 'https://github.com/cdmacfadyen/medpicpy',   
    keywords = ['medical-imaging', 'python', 'computer-vision', 'machine-learning'],  
    install_requires=[            
        'cycler',
        'importlib-metadata',
        'kiwisolver',
        'Mako',
        'Markdown',
        'MarkupSafe',
        'matplotlib',
        'numpy',
        'opencv-python',
        'pandas',
        'pdoc3',
        'Pillow',
        'pyparsing',
        'python-dateutil',
        'pytz',
        'SimpleITK',
        'six',
        'zipp'
    ],
classifiers=[
    'Development Status :: 5 - Production/Stable',      
    'Intended Audience :: Developers',      
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      
    ],
)