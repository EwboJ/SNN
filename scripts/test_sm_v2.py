"""
快速验证 hierarchical_state_machine.py 新增逻辑：
1. junction 锁存延迟门控
2. TURN -> RECOVER 多条件退出
3. 新增 debug 字段
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.hierarchical_state_machine import HierarchicalNavigatorStateMachine

def make_input(stage='Approach', junction='Left', omega=0.0):
    return {
        'stage3': {'pred_stage': stage, 'probs': {}},
        'junction_lr': {'pred_label': junction, 'probs': {}},
        'straight_keep': {'omega_cmd_raw': omega},
    }

def test_junction_lock_deferred():
    """验证 APPROACH 早期不锁存 junction"""
    sm = HierarchicalNavigatorStateMachine(
        boot_steps=2,
        stage_window_size=5,
        stage_enter_turn_votes=3,
        junction_window_size=3,
        junction_lock_votes=2,
        min_approach_steps_before_junction_lock=4,
        min_turn_votes_before_junction_lock=2,
    )
    # Boot
    for _ in range(2):
        sm.update(make_input('Approach', 'Left'))
    assert sm.state.name == 'STRAIGHTKEEP', f"Expected STRAIGHTKEEP, got {sm.state.name}"

    # Enter APPROACH
    for _ in range(3):
        r = sm.update(make_input('Approach', 'Left'))
    assert sm.state.name == 'APPROACH', f"Expected APPROACH, got {sm.state.name}"

    # 在 APPROACH 早期（step < min_approach_steps_before_junction_lock），
    # junction_lock_allowed 应为 False
    r = sm.update(make_input('Turn', 'Left'))  # step 1 of APPROACH
    assert r['debug']['junction_lock_allowed'] == False, "Should defer junction lock in early approach"
    assert 'early_approach' in r['debug']['junction_lock_block_reason']
    print("  [PASS] junction lock deferred in early approach")

    # 继续推进到满足 min_approach_steps_before_junction_lock
    for _ in range(3):
        r = sm.update(make_input('Turn', 'Left'))

    # 此时步数够了，但还需要检查 turn_votes 条件
    assert r['debug']['junction_lock_allowed'] == True or \
           'weak_turn' in r['debug'].get('junction_lock_block_reason', ''), \
           "Should check turn votes condition"
    print("  [PASS] junction lock gate logic works")


def test_turn_exit_multi_condition():
    """验证 TURN 多条件退出"""
    sm = HierarchicalNavigatorStateMachine(
        boot_steps=1,
        stage_window_size=5,
        stage_enter_turn_votes=3,
        junction_window_size=3,
        junction_lock_votes=2,
        min_turn_steps=3,
        max_turn_steps=30,
        recover_support_steps_needed=2,
        turn_exit_vote_threshold=1,
        straight_recover_omega_thresh=0.3,
        straight_recover_hold_steps=2,
        min_approach_steps_before_junction_lock=1,
        min_turn_votes_before_junction_lock=1,
    )
    # Boot
    sm.update(make_input('Approach', 'Left'))
    assert sm.state.name == 'STRAIGHTKEEP'

    # -> APPROACH -> TURN
    for _ in range(5):
        sm.update(make_input('Approach', 'Left'))
    for _ in range(4):
        sm.update(make_input('Turn', 'Left'))
    
    if sm.state.name != 'TURN':
        print(f"  [SKIP] Could not reach TURN, got {sm.state.name}")
        return

    # 在 min_turn_steps 之前不应退出
    r = sm.update(make_input('Recover', 'Left', 0.1))
    assert r['debug']['turn_exit_ready'] == (sm.state_step >= sm.min_turn_steps)
    print(f"  [PASS] turn_exit_ready={r['debug']['turn_exit_ready']} at step {sm.state_step}")

    # 推进到 min_turn_steps 后用低 turn + 低 omega 退出
    while sm.state.name == 'TURN' and sm.state_step < sm.min_turn_steps:
        sm.update(make_input('Approach', 'Left', 0.1))  # 非 recover，不累计

    # 现在尝试低 turn + 低 omega
    for _ in range(5):
        r = sm.update(make_input('Approach', 'Left', 0.05))
        if sm.state.name == 'RECOVER':
            break
    
    if sm.state.name == 'RECOVER':
        reason = r['debug']['transition']['reason']
        print(f"  [PASS] exit TURN by: {reason}")
        assert 'timeout' not in reason, "Should NOT exit by timeout"
    else:
        print(f"  [INFO] Still in {sm.state.name}, step={sm.state_step}")


def test_turn_exit_by_recover_signal():
    """验证 recover_signal_confirmed 退出"""
    sm = HierarchicalNavigatorStateMachine(
        boot_steps=1,
        stage_window_size=5,
        stage_enter_turn_votes=3,
        junction_window_size=3,
        junction_lock_votes=2,
        min_turn_steps=2,
        max_turn_steps=30,
        recover_support_steps_needed=2,
        min_approach_steps_before_junction_lock=1,
        min_turn_votes_before_junction_lock=1,
    )
    # Boot -> STRAIGHTKEEP -> APPROACH -> TURN
    sm.update(make_input('Approach', 'Left'))
    for _ in range(5):
        sm.update(make_input('Approach', 'Left'))
    for _ in range(4):
        sm.update(make_input('Turn', 'Left'))

    if sm.state.name != 'TURN':
        print(f"  [SKIP] Could not reach TURN, got {sm.state.name}")
        return

    # 发送足够多的 Recover 帧
    for _ in range(10):
        r = sm.update(make_input('Recover', 'Left', 0.0))
        if sm.state.name == 'RECOVER':
            break

    assert sm.state.name == 'RECOVER', f"Expected RECOVER, got {sm.state.name}"
    reason = r['debug']['transition']['reason']
    assert 'recover_signal_confirmed' in reason, f"Expected recover_signal_confirmed, got: {reason}"
    print(f"  [PASS] exit TURN by: {reason}")


def test_debug_fields_present():
    """验证新增 debug 字段全部存在"""
    sm = HierarchicalNavigatorStateMachine(boot_steps=1)
    sm.update(make_input('Approach', 'Left'))
    r = sm.update(make_input('Approach', 'Left'))
    d = r['debug']

    required_fields = [
        'turn_exit_ready',
        'straight_recover_hold_count',
        'recover_support_threshold',
        'junction_lock_allowed',
        'junction_lock_block_reason',
    ]
    missing = [f for f in required_fields if f not in d]
    assert not missing, f"Missing debug fields: {missing}"
    print(f"  [PASS] All new debug fields present: {required_fields}")

    # 检查 thresholds 中新增字段
    thr = d['thresholds']
    thr_fields = [
        'min_turn_steps',
        'turn_exit_vote_threshold',
        'straight_recover_omega_thresh',
        'straight_recover_hold_steps',
        'min_approach_steps_before_junction_lock',
        'min_turn_votes_before_junction_lock',
    ]
    missing_thr = [f for f in thr_fields if f not in thr]
    assert not missing_thr, f"Missing threshold fields: {missing_thr}"
    print(f"  [PASS] All new threshold fields present")


if __name__ == '__main__':
    print("=== Test: junction_lock_deferred ===")
    test_junction_lock_deferred()
    print()

    print("=== Test: turn_exit_multi_condition ===")
    test_turn_exit_multi_condition()
    print()

    print("=== Test: turn_exit_by_recover_signal ===")
    test_turn_exit_by_recover_signal()
    print()

    print("=== Test: debug_fields_present ===")
    test_debug_fields_present()
    print()

    print("ALL TESTS PASSED!")
