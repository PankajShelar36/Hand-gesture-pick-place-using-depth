from setuptools import find_packages, setup

package_name = 'pick_place'

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
    maintainer='ubuntu',
    maintainer_email='samzach333@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'red_box_detector = pick_place.red_box_detector:main',
            'pick_node = pick_place.pick_node:main',
            'spawn_scene = pick_place.spawn_scene:main',
            'any_object_detector = pick_place.any_object_detector:main',
            'new_pick_node = pick_place.new_pick_node:main',
            'pick_table = pick_place.pick_table:main',
            'red_detector_rs = pick_place.red_detector_rs:main',
            'pick_node_rs = pick_place.pick_node_rs:main',
            'any_pose_detector_rs = pick_place.any_pose_detector_rs:main',
            'pick_node_any_pose = pick_place.pick_node_any_pose:main',
            'pointcloud_collision_monitor = pick_place.pointcloud_collision_monitor:main',
            'voice_color_detector_rs = pick_place.voice_color_detector_rs:main',
            'voice_pick_node = pick_place.voice_pick_node:main',
            'handover_pick_node = pick_place.handover_pick_node:main',
            'gesture_handover_pick_node = pick_place.gesture_handover_pick_node:main',
            'gesture_place_node = pick_place.gesture_place_node:main',
        ],
    },
)
