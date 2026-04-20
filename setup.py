from setuptools import setup
import os
from glob import glob

package_name = 'catrun'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
        ('share/' + package_name + '/map',
            glob('map/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cc',
    maintainer_email='mojiking8888@gmail.com',
    description='Cat robot package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'cat_detection = catrun.cat_detection:main',
            'motor_control = catrun.motor_control:main',
            'navigation = catrun.navigation:main',
        ],
    },
)