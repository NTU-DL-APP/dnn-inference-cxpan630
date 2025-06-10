import tensorflow as tf
import numpy as np
import os

def create_fashion_mnist_model():
    """建立並訓練一個簡單的Fashion-MNIST分類模型"""
    
    print("🔄 正在載入Fashion-MNIST資料集...")
    
    # 載入Fashion-MNIST資料集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # 資料預處理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"訓練資料形狀: {x_train.shape}")
    print(f"測試資料形狀: {x_test.shape}")
    
    # 建立模型
    print("\n🏗️ 正在建立模型...")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='dense'),
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(10, activation='softmax', name='dense_2')
    ])
    
    # 編譯模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 顯示模型架構
    print("\n📋 模型架構:")
    model.summary()
    
    # 訓練模型
    print("\n🚀 開始訓練模型...")
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # 評估模型
    print("\n📊 評估模型性能...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"測試準確率: {test_accuracy:.4f}")
    
    # 儲存模型
    model_path = 'fashion_mnist.h5'
    model.save(model_path)
    print(f"\n💾 模型已儲存至: {model_path}")
    
    return model, test_accuracy

def setup_data_folder():
    """設置資料夾結構"""
    os.makedirs('model', exist_ok=True)
    os.makedirs('data/fashion', exist_ok=True)
    print("📁 資料夾結構已建立")

if __name__ == "__main__":
    print("🎯 Fashion-MNIST 模型建立程式")
    print("=" * 50)
    
    # 設置資料夾
    setup_data_folder()
    
    # 檢查是否已有模型
    if os.path.exists('fashion_mnist.h5'):
        choice = input("已存在 fashion_mnist.h5 檔案，要重新訓練嗎？(y/n): ").lower()
        if choice != 'y':
            print("使用現有模型檔案")
            exit()
    
    try:
        # 建立和訓練模型
        model, accuracy = create_fashion_mnist_model()
        
        print("\n✅ 模型建立完成！")
        print(f"最終測試準確率: {accuracy:.4f}")
        print("\n📝 下一步:")
        print("1. 執行 'python convert_model.py' 轉換模型")
        print("2. 執行 'python model_test.py' 測試推理")
        
    except Exception as e:
        print(f"❌ 建立模型時發生錯誤: {e}")
        print("請檢查是否已安裝 TensorFlow: pip install tensorflow")