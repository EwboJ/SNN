#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Humble launch：真实小车第一版层级导航闭环原型。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # 可从命令行覆盖的参数
    config_path_arg = DeclareLaunchArgument(
        "config_path",
        default_value="configs/hierarchical_nav_robot_v1.yaml",
        description="Path to hierarchical navigation runtime config yaml.",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/color/image_raw",
        description="Input camera image topic.",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="Output cmd_vel topic.",
    )
    # 状态和调试 topic 也支持 launch 命令行覆盖，便于多实例或测试隔离。
    state_topic_arg = DeclareLaunchArgument(
        "state_topic",
        default_value="/nav/state",
        description="Output navigation state topic.",
    )
    debug_topic_arg = DeclareLaunchArgument(
        "debug_topic",
        default_value="/nav/debug",
        description="Output navigation debug topic.",
    )

    config_path = LaunchConfiguration("config_path")
    image_topic = LaunchConfiguration("image_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    state_topic = LaunchConfiguration("state_topic")
    debug_topic = LaunchConfiguration("debug_topic")

    runtime_node = Node(
        package="snn_nav_ros",
        executable="hierarchical_nav_runtime",
        name="hierarchical_nav_runtime",
        output="screen",
        parameters=[
            {
                "config_path": config_path,
                "image_topic": image_topic,
                "cmd_vel_topic": cmd_vel_topic,
                "state_topic": state_topic,
                "debug_topic": debug_topic,
            }
        ],
    )

    return LaunchDescription(
        [
            config_path_arg,
            image_topic_arg,
            cmd_vel_topic_arg,
            state_topic_arg,
            debug_topic_arg,
            runtime_node,
        ]
    )

