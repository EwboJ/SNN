from glob import glob
from setuptools import setup

package_name = "snn_nav_ros"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        # 向 ament index 注册包名
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # 安装 package.xml 供 ROS2 元数据发现
        ("share/" + package_name, ["package.xml"]),
        # 安装 launch 文件，供 ros2 launch 调用
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SNN Nav Maintainer",
    maintainer_email="maintainer@example.com",
    description="ROS2 hierarchical navigation online runtime package for SNN-based navigation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hierarchical_nav_runtime = snn_nav_ros.hierarchical_nav_runtime_node:main",
        ],
    },
)
