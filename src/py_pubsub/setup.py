from setuptools import setup

package_name = 'py_pubsub'
submodules ="py_pubsub/submodule"
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name,submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eric',
    maintainer_email='t108650033@ntut.org.tw',
    description='Examples of minimal publisher/subscriber using rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'talker = py_pubsub.publisher_member_function:main',
		'listener = py_pubsub.subscriber_member_function:main',
		],
    },
)
