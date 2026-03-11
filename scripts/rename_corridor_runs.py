import os
import re

# 数据目录
DATA_DIR = './data/corridor_balanced'

# 新命名规则：J{junction}_{direction}_r{采集编号}
# 原命名格式：left1_bag1/right2_bag3/straight3_bag2
# direction: left/right/straight
# junction: 数字
# bag: 采集编号

def rename_runs(data_dir):
    pattern = re.compile(r'^(left|right|straight)(\d+)_bag(\d+)$')
    for name in os.listdir(data_dir):
        old_path = os.path.join(data_dir, name)
        if not os.path.isdir(old_path):
            continue
        m = pattern.match(name)
        if not m:
            print(f'Skip: {name}')
            continue
        direction, junction, bag = m.groups()
        new_name = f'J{junction}_{direction}_r{int(bag):02d}'
        new_path = os.path.join(data_dir, new_name)
        if os.path.exists(new_path):
            print(f'目标已存在: {new_name}, 跳过')
            continue
        print(f'{name} -> {new_name}')
        os.rename(old_path, new_path)

if __name__ == '__main__':
    rename_runs(DATA_DIR)
    print('重命名完成。')
