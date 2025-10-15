import pandas as pd
import numpy as np
import re
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input, Bidirectional, Conv1D, GlobalMaxPooling1D, \
    concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

print("加载数据并提取特征...")

# 1. 数据加载
df = pd.read_csv('cybersecurity_attacks.csv')
print(f"数据形状: {df.shape}")

# 只保留需要的列
df = df[['Payload Data', 'Attack Type']]
print(f"攻击类型分布:\n{df['Attack Type'].value_counts()}")


# 2. 增强的特征工程
class AdvancedFeatureExtractor:
    def __init__(self):
        # 编码模式
        self.encoding_patterns = {
            'base64': r'[A-Za-z0-9+/]{4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?',
            'hex': r'[0-9A-Fa-f]{4,}',
            'url_encoded': r'%[0-9A-Fa-f]{2}',
            'special_sequences': r'[^\w\s]'
        }

        # 可疑关键词模式
        self.suspicious_patterns = [
            r'\b(exec|cmd|bash|sh|powershell|wget|curl|ftp)\b',
            r'\b(select|insert|update|delete|drop|union|where)\b',
            r'\b(script|alert|eval|document|window)\b',
            r'\b(admin|root|password|passwd|login)\b',
            r'\b(system|shell|runtime|process)\b'
        ]

    def extract_features(self, text):
        """提取高级特征"""
        if pd.isna(text):
            text = ""
        else:
            text = str(text)

        features = {}

        # 1. 文本统计特征
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_diversity'] = len(set(text)) / max(1, len(text))

        # 2. 编码特征检测
        features['base64_count'] = len(re.findall(self.encoding_patterns['base64'], text))
        features['hex_count'] = len(re.findall(self.encoding_patterns['hex'], text))
        features['url_encoded_count'] = len(re.findall(self.encoding_patterns['url_encoded'], text))
        features['special_char_ratio'] = len(re.findall(self.encoding_patterns['special_sequences'], text)) / max(1,
                                                                                                                  len(text))

        # 3. 可疑模式检测
        suspicious_score = 0
        for pattern in self.suspicious_patterns:
            suspicious_score += len(re.findall(pattern, text, re.IGNORECASE))
        features['suspicious_score'] = suspicious_score

        # 4. 熵特征（检测随机性）
        features['entropy'] = self.calculate_entropy(text)

        # 5. 结构特征
        features['has_whitespace'] = 1 if re.search(r'\s', text) else 0
        features['has_digits'] = 1 if re.search(r'\d', text) else 0
        features['has_uppercase'] = 1 if re.search(r'[A-Z]', text) else 0
        features['has_lowercase'] = 1 if re.search(r'[a-z]', text) else 0

        return features

    def calculate_entropy(self, text):
        """计算文本熵"""
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = text.count(chr(x)) / len(text)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
        return entropy


# 3. 应用特征提取器
print("\n提取高级特征...")
feature_extractor = AdvancedFeatureExtractor()

# 提取特征
feature_results = []
for payload in df['Payload Data']:
    feature_results.append(feature_extractor.extract_features(payload))

# 将特征添加到DataFrame
feature_df = pd.DataFrame(feature_results)
df = pd.concat([df, feature_df], axis=1)

print(f"提取的特征数量: {len(feature_df.columns)}")
print("特征描述:")
print(feature_df.describe())

# 4. 数据预处理
print("\n数据预处理...")

# 处理标签
label_encoder = LabelEncoder()
df['attack_label'] = label_encoder.fit_transform(df['Attack Type'])
print(f"标签编码: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


# 文本预处理函数
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # 清理文本但保留特殊字符（对于安全分析很重要）
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


# 准备文本数据
texts = df['Payload Data'].apply(preprocess_text).tolist()

# 文本分词和序列化
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>", filters='')
tokenizer.fit_on_texts(texts)

text_sequences = tokenizer.texts_to_sequences(texts)
max_sequence_length = 200
X_text = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# 准备数值特征
numeric_features = [
    'text_length', 'word_count', 'char_diversity', 'base64_count',
    'hex_count', 'url_encoded_count', 'special_char_ratio',
    'suspicious_score', 'entropy', 'has_whitespace', 'has_digits',
    'has_uppercase', 'has_lowercase'
]
X_numeric = df[numeric_features].values

# 标准化数值特征
numeric_scaler = StandardScaler()
X_numeric = numeric_scaler.fit_transform(X_numeric)

# 目标变量
y = to_categorical(df['attack_label'])

print(f"文本特征形状: {X_text.shape}")
print(f"数值特征形状: {X_numeric.shape}")
print(f"标签形状: {y.shape}")
print(f"词汇表大小: {len(tokenizer.word_index)}")

# 5. 构建增强的LSTM-CNN混合模型
print("\n构建增强的LSTM-CNN混合模型...")


def create_enhanced_lstm_model(vocab_size, max_sequence_length, numeric_dim, num_classes, embedding_dim=128):
    """创建增强的LSTM-CNN混合模型"""

    # 文本输入
    text_input = Input(shape=(max_sequence_length,), name='text_input')

    # 嵌入层
    embedding = Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        name='embedding'
    )(text_input)

    # 并行处理路径1: CNN用于捕捉局部模式
    conv1 = Conv1D(64, 3, activation='relu', padding='same', name='conv1d_3')(embedding)
    conv1 = Conv1D(64, 3, activation='relu', padding='same', name='conv1d_3_2')(conv1)
    conv1_pool = GlobalMaxPooling1D(name='global_pool_3')(conv1)

    conv2 = Conv1D(64, 5, activation='relu', padding='same', name='conv1d_5')(embedding)
    conv2 = Conv1D(64, 5, activation='relu', padding='same', name='conv1d_5_2')(conv2)
    conv2_pool = GlobalMaxPooling1D(name='global_pool_5')(conv2)

    # 并行处理路径2: LSTM用于捕捉序列依赖
    lstm_layer = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        name='bi_lstm_1'
    )(embedding)
    lstm_layer = Bidirectional(
        LSTM(32, dropout=0.3, recurrent_dropout=0.3),
        name='bi_lstm_2'
    )(lstm_layer)

    # 数值特征输入
    numeric_input = Input(shape=(numeric_dim,), name='numeric_input')
    numeric_dense = Dense(64, activation='relu', name='numeric_dense_1')(numeric_input)
    numeric_dense = Dropout(0.3, name='numeric_dropout_1')(numeric_dense)
    numeric_dense = Dense(32, activation='relu', name='numeric_dense_2')(numeric_dense)

    # 特征融合
    combined = concatenate([lstm_layer, conv1_pool, conv2_pool, numeric_dense], name='feature_fusion')

    # 注意力机制（简化版）
    attention = Dense(combined.shape[-1], activation='tanh', name='attention_dense')(combined)
    attention_weights = Dense(1, activation='softmax', name='attention_weights')(attention)

    # 分类头
    classifier = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='classifier_1')(combined)
    classifier = Dropout(0.4, name='classifier_dropout_1')(classifier)
    classifier = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='classifier_2')(
        classifier)
    classifier = Dropout(0.3, name='classifier_dropout_2')(classifier)
    classifier = Dense(32, activation='relu', name='classifier_3')(classifier)

    # 输出层
    output = Dense(num_classes, activation='softmax', name='output')(classifier)

    # 创建模型
    model = Model(
        inputs=[text_input, numeric_input],
        outputs=output,
        name='enhanced_lstm_model'
    )

    return model


# 创建模型
vocab_size = len(tokenizer.word_index)
numeric_dim = X_numeric.shape[1]
num_classes = y.shape[1]

model = create_enhanced_lstm_model(vocab_size, max_sequence_length, numeric_dim, num_classes)

# 编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# 6. 数据集划分 (7:1:2)
print("\n划分数据集...")

# 首先划分训练集和临时集
X_text_train, X_text_temp, X_numeric_train, X_numeric_temp, y_train, y_temp = train_test_split(
    X_text, X_numeric, y, test_size=0.3, random_state=42, stratify=df['attack_label']
)

# 然后划分验证集和测试集
X_text_val, X_text_test, X_numeric_val, X_numeric_test, y_val, y_test = train_test_split(
    X_text_temp, X_numeric_temp, y_temp, test_size=0.6667, random_state=42, stratify=df.iloc[temp_idx]['attack_label']
)

print(f"训练集大小: {len(X_text_train)}")
print(f"验证集大小: {len(X_text_val)}")
print(f"测试集大小: {len(X_text_test)}")

# 准备输入数据
train_inputs = {
    'text_input': X_text_train,
    'numeric_input': X_numeric_train
}

val_inputs = {
    'text_input': X_text_val,
    'numeric_input': X_numeric_val
}

test_inputs = {
    'text_input': X_text_test,
    'numeric_input': X_numeric_test
}

# 7. 训练模型
print("\n开始训练模型...")

# 设置回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 训练模型
history = model.fit(
    train_inputs, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(val_inputs, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 8. 模型评估
print("\n评估模型...")

# 预测
y_pred = model.predict(test_inputs)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 计算准确率
test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"测试集准确率: {test_accuracy:.4f}")

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_true_classes, y_pred_classes,
                            target_names=label_encoder.classes_))

# 9. 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('攻击类型分类混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 10. 训练过程可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.tight_layout()
plt.show()

# 11. 特征重要性分析
print("\n=== 特征重要性分析 ===")
feature_importance = {}
for i, feature in enumerate(numeric_features):
    correlation = np.corrcoef(df[feature], df['attack_label'])[0, 1]
    feature_importance[feature] = abs(correlation)

# 按重要性排序
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
print("特征重要性排序:")
for feature, importance in sorted_features:
    print(f"  {feature}: {importance:.4f}")

# 12. 保存模型和工具
print("\n保存模型和预处理工具...")
model.save('enhanced_lstm_cybersecurity_model.h5')

import joblib

joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(numeric_scaler, 'numeric_scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(feature_extractor, 'feature_extractor.pkl')

print("模型和工具保存完成!")


# 13. 创建实时检测函数
def real_time_attack_detection(text):
    """实时攻击检测函数"""
    # 预处理文本
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')

    # 提取特征
    features = feature_extractor.extract_features(text)
    feature_vector = np.array([[features[col] for col in numeric_features]])
    feature_vector = numeric_scaler.transform(feature_vector)

    # 预测
    prediction = model.predict({
        'text_input': padded_sequence,
        'numeric_input': feature_vector
    })

    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    confidence = float(np.max(prediction))

    return {
        'predicted_attack_type': predicted_label,
        'confidence': confidence,
        'suspicious_score': features['suspicious_score'],
        'encoding_features': features['base64_count'] + features['hex_count'] + features['url_encoded_count'],
        'threat_level': 'HIGH' if features['suspicious_score'] > 2 else 'MEDIUM' if features[
                                                                                        'suspicious_score'] > 0 else 'LOW'
    }


# 测试实时检测
print("\n=== 实时攻击检测测试 ===")
test_samples = [
    "Qui natus odio asperiores nam. Optio nobis iusto accusamus ad perferendis esse at.",
    "exec cmd /c dir && ping 127.0.0.1",
    "TWFpb3JlcyBwb3NzaW11cw== select * from users where id=1",
    "alert(document.cookie); eval('malicious code')",
    "normal text without any suspicious patterns"
]

for i, sample in enumerate(test_samples):
    result = real_time_attack_detection(sample)
    print(f"\n样本 {i + 1}:")
    print(f"  内容: {sample[:50]}...")
    print(f"  预测攻击类型: {result['predicted_attack_type']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  可疑分数: {result['suspicious_score']}")
    print(f"  威胁等级: {result['threat_level']}")

print("\n=== 模型训练和评估完成 ===")