from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='MoveAsAppleVisionPro',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'mediapipe',
        'torch',
        'screeninfo',
        'matplotlib',
        'joblib',
        'pyautogui',
        'scikit-learn',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'move-as-apple-vision-pro=move_as_apple_vision_pro.main:main',
        ],
    },
    author='Yubo Dong',
    author_email='22321287@zju.edu.cn',
    description='A project to move the mouse as Apple Vision Pro',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cnsdqd-dyb/operate_as_apple_vision_pro',
    # ...
)
