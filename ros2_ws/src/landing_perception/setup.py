from setuptools import setup
import os
from glob import glob

package_name = 'landing_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shreyansh',
    maintainer_email='shreyansh@todo.todo',
    description='Perceptron-based perception and landing controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = landing_perception.vision_node:main',
            'perceptron_node = landing_perception.perceptron_node:main',
            'planner_node = landing_perception.planner_node:main',
            'controller_node = landing_perception.controller_node:main',
        ],
    },
)