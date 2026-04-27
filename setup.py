from setuptools import setup
from glob import glob
import os

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
            glob('map/*.pgm') + glob('map/*.yaml')),
	    (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cc',
    maintainer_email='mojiking8888@gmail.com',
    description='Cat robot package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            
            'motor_control = catrun.motor_control:main',
            'navigation = catrun.navigation:main',
            'flee_behavior = catrun.flee_behavior:main',

            'camera_node = catrun.camera_node:main',
            'web_stream = catrun.web_stream:main',
            'cat_detector = catrun.cat_detector:main',
            'seek_cat = catrun.seek_cat:main',
        ],
    },
)
