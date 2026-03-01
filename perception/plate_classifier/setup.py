from setuptools import setup, find_packages

setup(
    name="light_plate_classifier",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True, # 确保包含 models 文件夹里的 onnx 文件
    package_data={
        '': ['models/*.onnx'],
    },
    install_requires=[
        "numpy",
        "opencv-python",
        "onnxruntime" # 核心运行库
    ]
)
