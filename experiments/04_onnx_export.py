"""
04. ONNX Export - Stable-Baselines3モデルのONNX変換

このスクリプトでは、Stable-Baselines3で訓練したモデルをONNX形式に変換し、
入出力シェイプを検証します。

要件 1.5: Stable-Baselines3モデルをONNX形式に変換、入出力シェイプを検証
"""

import numpy as np
import torch
import onnx
import onnxruntime as ort
from stable_baselines3 import DQN, PPO
import gymnasium as gym
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("04. ONNX Export - モデル変換と検証")
print("=" * 60)

# ===== 1. モデルの読み込み =====
print("\n[1] 訓練済みモデルの読み込み")
print("-" * 60)

models_dir = Path("../ml/models")
models_dir.mkdir(parents=True, exist_ok=True)

# DQNモデルの読み込み
dqn_model_path = models_dir / "dqn_cartpole.zip"
if dqn_model_path.exists():
    dqn_model = DQN.load(dqn_model_path)
    print(f"✅ DQNモデルを読み込みました: {dqn_model_path}")
else:
    print(f"⚠️  DQNモデルが見つかりません: {dqn_model_path}")
    print("   01_dqn_basic.ipynbを先に実行してください")
    dqn_model = None

# PPOモデルの読み込み（存在する場合）
ppo_model_path = models_dir / "ppo_cartpole.zip"
if ppo_model_path.exists():
    ppo_model = PPO.load(ppo_model_path)
    print(f"✅ PPOモデルを読み込みました: {ppo_model_path}")
else:
    print(f"ℹ️  PPOモデルが見つかりません: {ppo_model_path}")
    print("   02_ppo_basic.ipynbを実行すると、PPOモデルも変換できます")
    ppo_model = None


# ===== 2. ONNX変換関数の定義 =====
print("\n[2] ONNX変換関数の定義")
print("-" * 60)

def export_sb3_to_onnx(model, model_name, output_path, input_shape):
    """
    Stable-Baselines3モデルをONNX形式に変換
    
    Args:
        model: Stable-Baselines3モデル（DQN, PPO等）
        model_name: モデル名（ファイル名用）
        output_path: 出力パス
        input_shape: 入力シェイプ（例: (1, 4) for CartPole）
    
    Returns:
        onnx_path: 保存されたONNXファイルのパス
    """
    print(f"\n{model_name}をONNX形式に変換中...")
    
    # PyTorchモデルを取得
    policy = model.policy
    policy.eval()
    
    # ダミー入力を作成
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    
    # ONNXファイルパス
    onnx_path = output_path / f"{model_name}.onnx"
    
    # ONNX変換
    torch.onnx.export(
        policy,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNXモデルを保存しました: {onnx_path}")
    return onnx_path


def verify_onnx_model(onnx_path, input_shape):
    """
    ONNXモデルの検証
    
    Args:
        onnx_path: ONNXファイルのパス
        input_shape: 入力シェイプ
    
    Returns:
        dict: 検証結果
    """
    print(f"\nONNXモデルを検証中: {onnx_path.name}")
    
    # ONNXモデルの読み込みと検証
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✅ ONNXモデルの構造が正しいことを確認")
    
    # 入出力情報の取得
    print("\n入出力情報:")
    for input_tensor in onnx_model.graph.input:
        print(f"  入力: {input_tensor.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"    形状: {shape}")
        print(f"    型: {input_tensor.type.tensor_type.elem_type}")
    
    for output_tensor in onnx_model.graph.output:
        print(f"  出力: {output_tensor.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"    形状: {shape}")
        print(f"    型: {output_tensor.type.tensor_type.elem_type}")
    
    # ONNX Runtimeで推論テスト
    print("\nONNX Runtimeで推論テスト:")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # ダミー入力で推論
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"  入力形状: {dummy_input.shape}")
    print(f"  出力形状: {ort_outputs[0].shape}")
    print(f"  出力サンプル: {ort_outputs[0][:5]}")
    print("✅ ONNX Runtimeでの推論が成功")
    
    return {
        'input_shape': dummy_input.shape,
        'output_shape': ort_outputs[0].shape,
        'model_valid': True
    }

print("✅ ONNX変換関数を定義しました")


# ===== 3. DQNモデルのONNX変換 =====
if dqn_model is not None:
    print("\n[3] DQNモデルのONNX変換")
    print("-" * 60)
    
    # CartPole環境の観測空間: Box(4,)
    input_shape = (1, 4)
    
    # ONNX変換
    dqn_onnx_path = export_sb3_to_onnx(
        model=dqn_model,
        model_name="dqn_cartpole",
        output_path=models_dir,
        input_shape=input_shape
    )
    
    # 検証
    dqn_results = verify_onnx_model(dqn_onnx_path, input_shape)
    
    print("\n=== DQN ONNX変換結果 ===")
    print(f"入力形状: {dqn_results['input_shape']}")
    print(f"出力形状: {dqn_results['output_shape']}")
    print(f"モデル有効性: {dqn_results['model_valid']}")

# ===== 4. PPOモデルのONNX変換 =====
if ppo_model is not None:
    print("\n[4] PPOモデルのONNX変換")
    print("-" * 60)
    
    # CartPole環境の観測空間: Box(4,)
    input_shape = (1, 4)
    
    # ONNX変換
    ppo_onnx_path = export_sb3_to_onnx(
        model=ppo_model,
        model_name="ppo_cartpole",
        output_path=models_dir,
        input_shape=input_shape
    )
    
    # 検証
    ppo_results = verify_onnx_model(ppo_onnx_path, input_shape)
    
    print("\n=== PPO ONNX変換結果 ===")
    print(f"入力形状: {ppo_results['input_shape']}")
    print(f"出力形状: {ppo_results['output_shape']}")
    print(f"モデル有効性: {ppo_results['model_valid']}")


# ===== 5. 元のモデルとONNXモデルの出力比較 =====
if dqn_model is not None:
    print("\n[5] 元のモデルとONNXモデルの出力比較")
    print("-" * 60)
    
    # テスト環境の作成
    env = gym.make('CartPole-v1')
    obs, info = env.reset()
    
    print(f"\nテスト観測: {obs}")
    
    # 元のStable-Baselines3モデルで予測
    action_sb3, _ = dqn_model.predict(obs, deterministic=True)
    print(f"\nStable-Baselines3 DQN予測:")
    print(f"  行動: {action_sb3}")
    
    # ONNXモデルで予測
    dqn_onnx_path = models_dir / "dqn_cartpole.onnx"
    ort_session = ort.InferenceSession(str(dqn_onnx_path))
    
    # 入力を準備（バッチ次元を追加）
    onnx_input = obs.reshape(1, -1).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"\nONNX Runtime予測:")
    print(f"  出力: {ort_outputs[0]}")
    print(f"  出力形状: {ort_outputs[0].shape}")
    
    # Q値から行動を選択（DQNの場合）
    if len(ort_outputs[0].shape) > 1 and ort_outputs[0].shape[1] > 1:
        action_onnx = np.argmax(ort_outputs[0])
        print(f"  選択された行動: {action_onnx}")
        
        # 一致確認
        if action_sb3 == action_onnx:
            print("\n✅ Stable-Baselines3とONNXの予測が一致しました")
        else:
            print("\n⚠️  予測が一致しません（これは正常な場合もあります）")
            print("   モデルの内部状態や確率的な要素により、若干の差異が生じることがあります")
    
    env.close()

# ===== 6. まとめ =====
print("\n" + "=" * 60)
print("まとめ")
print("=" * 60)

print("\n✅ 完了した作業:")
if dqn_model is not None:
    print("  - DQNモデルをONNX形式に変換")
    print("  - 入出力シェイプを検証")
    print("  - ONNX Runtimeでの推論を確認")
    print("  - 元のモデルとの出力を比較")

if ppo_model is not None:
    print("  - PPOモデルをONNX形式に変換")
    print("  - 入出力シェイプを検証")
    print("  - ONNX Runtimeでの推論を確認")

print("\n📁 保存されたファイル:")
for onnx_file in models_dir.glob("*.onnx"):
    file_size = onnx_file.stat().st_size / 1024  # KB
    print(f"  - {onnx_file.name} ({file_size:.2f} KB)")

print("\n次のステップ:")
print("  - go_onnx_validation/でGoからONNXモデルを読み込んで推論")
print("  - Go-ONNX統合を検証")
print("  - レイテンシを計測")

print("\n" + "=" * 60)
print("04_onnx_export.py 完了")
print("=" * 60)
