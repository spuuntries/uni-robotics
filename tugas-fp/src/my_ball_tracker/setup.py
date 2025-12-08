from setuptools import find_packages, setup

package_name = 'my_ball_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kek',
    maintainer_email='kek@spuun.art',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'my_tracker = my_ball_tracker.ball_tracker_node:main',
                'cube_collector = my_ball_tracker.cube_collector_node:main',
                'cube_spawner = my_ball_tracker.cube_spawner:main',
            ],
    },
)
