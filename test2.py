import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置 - 强制使用CPU
device = torch.device('cpu')
print(f"Using device: {device}")


class SecurityDataset(Dataset):
    def __init__(self, texts, structured_features, labels, tokenizer, max_len=128):
        self.texts = texts
        self.structured_features = structured_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        structured_feature = self.structured_features[idx]
        label = self.labels[idx]

        # 处理文本数据
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'structured_features': torch.FloatTensor(structured_feature),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class HybridLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name, structured_feature_dim, num_classes, hidden_dim=128,
                 lstm_layers=1, dropout_rate=0.3):
        super(HybridLSTMClassifier, self).__init__()

        # BERT文本编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        # 冻结BERT的大部分层以减少计算量
        for param in list(self.bert.parameters())[:-20]:  # 只训练最后几层
            param.requires_grad = False

        # 文本LSTM编码器
        self.text_lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # 结构化特征处理（使用简单的全连接层而不是LSTM）
        self.struct_fc = nn.Sequential(
            nn.Linear(structured_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # 注意力机制
        self.text_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(dropout_rate)
        )

        # 分类器
        combined_features_size = hidden_dim * 2 + hidden_dim // 2  # 文本LSTM输出 + 结构化特征输出

        self.classifier = nn.Sequential(
            nn.Linear(combined_features_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, structured_features):
        # BERT编码
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state

        # 文本LSTM处理
        text_lstm_out, (text_hidden, _) = self.text_lstm(text_features)

        # 文本注意力机制
        attention_weights = torch.softmax(self.text_attention(text_lstm_out), dim=1)
        text_attended = torch.sum(attention_weights * text_lstm_out, dim=1)

        # 结构化特征处理
        struct_features = self.struct_fc(structured_features)

        # 特征融合
        combined_features = torch.cat([text_attended, struct_features], dim=1)
        combined_features = self.dropout(combined_features)

        # 分类
        logits = self.classifier(combined_features)

        return logits


class SecurityAttackClassifier:
    def __init__(self, bert_model_name='bert-base-uncased', max_len=128):
        self.bert_model_name = bert_model_name
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.device = device

    def load_and_preprocess_data(self, file_path='cybersecurity_attacks.csv'):
        """加载并预处理数据"""
        print("加载数据...")
        df = pd.read_csv(file_path)
        print(f"数据形状: {df.shape}")

        # 显示数据基本信息
        print(f"数据列名: {df.columns.tolist()}")
        if 'Attack Type' in df.columns:
            print(f"攻击类型分布:\n{df['Attack Type'].value_counts()}")
        print(f"缺失值统计:\n{df.isnull().sum()}")

        # 检查Payload Data字段
        if 'Payload Data' not in df.columns:
            print("警告: 数据集中没有找到'Payload Data'字段，使用第一个文本字段")
            # 尝试找到文本字段
            text_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'Attack Type']
            if text_columns:
                text_column = text_columns[0]
                print(f"使用 '{text_column}' 作为文本字段")
            else:
                raise ValueError("没有找到可用的文本字段")
        else:
            text_column = 'Payload Data'

        return self.preprocess_data(df, text_column=text_column)

    def preprocess_data(self, df, text_column='Payload Data', label_column='Attack Type'):
        """预处理数据"""
        print("预处理数据...")

        # 处理标签
        if label_column not in df.columns:
            # 如果没有标签列，创建一个虚拟标签
            print("警告: 没有找到标签列，创建虚拟标签")
            labels = np.zeros(len(df))
            self.label_encoder.fit(['Unknown'])
        else:
            labels = self.label_encoder.fit_transform(df[label_column])
        print(f"标签类别: {self.label_encoder.classes_}")

        # 选择结构化特征 - 基于题目描述的重要特征
        # 数值特征
        numeric_features = []
        potential_numeric = ['Packet Length', 'Anomaly Score', 'Severity Level', 'packet_length',
                             'anomaly_score', 'severity_level', 'Duration', 'duration',
                             'Source Port', 'source_port', 'Destination Port', 'destination_port']

        for feature in potential_numeric:
            if feature in df.columns:
                # 尝试转换为数值类型
                try:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    numeric_features.append(feature)
                except:
                    print(f"警告: 无法将特征 {feature} 转换为数值类型，跳过")

        # 类别特征
        categorical_features = []
        potential_categorical = ['Protocol', 'Flow Type', 'Action Taken', 'protocol',
                                 'flow_type', 'action_taken', 'Source IP', 'source_ip',
                                 'Destination IP', 'destination_ip']

        for feature in potential_categorical:
            if feature in df.columns and feature not in numeric_features:
                categorical_features.append(feature)

        print(f"使用的数值特征: {numeric_features}")
        print(f"使用的类别特征: {categorical_features}")

        # 处理数值特征
        if numeric_features:
            # 填充缺失值
            numeric_data = df[numeric_features].fillna(0).values
            # 标准化数值特征
            if len(numeric_data) > 0:
                numeric_data = self.scaler.fit_transform(numeric_data)
        else:
            numeric_data = np.zeros((len(df), 0))

        # 处理类别特征
        if categorical_features:
            # 填充缺失值
            categorical_df = df[categorical_features].fillna('unknown')
            # 对类别特征进行one-hot编码
            categorical_data = pd.get_dummies(categorical_df).values
        else:
            categorical_data = np.zeros((len(df), 0))

        # 合并结构化特征
        structured_features = np.hstack([numeric_data, categorical_data])

        print(f"结构化特征维度: {structured_features.shape}")
        print(f"文本特征数量: {len(df[text_column])}")

        return df[text_column].values, structured_features, labels

    def create_data_loaders(self, texts, structured_features, labels, batch_size=8):
        """创建数据加载器"""
        # 划分数据集 7:1:2
        X_text_temp, X_text_test, X_struct_temp, X_struct_test, y_temp, y_test = train_test_split(
            texts, structured_features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        X_text_train, X_text_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
            X_text_temp, X_struct_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp  # 0.125 * 0.8 = 0.1
        )

        print(f"训练集大小: {len(X_text_train)}")
        print(f"验证集大小: {len(X_text_val)}")
        print(f"测试集大小: {len(X_text_test)}")

        # 创建数据集
        train_dataset = SecurityDataset(X_text_train, X_struct_train, y_train, self.tokenizer, self.max_len)
        val_dataset = SecurityDataset(X_text_val, X_struct_val, y_val, self.tokenizer, self.max_len)
        test_dataset = SecurityDataset(X_text_test, X_struct_test, y_test, self.tokenizer, self.max_len)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(self, train_loader, val_loader, num_epochs=5, learning_rate=2e-5):
        """训练模型"""
        structured_feature_dim = train_loader.dataset[0]['structured_features'].shape[0]
        num_classes = len(self.label_encoder.classes_)

        print(f"结构化特征维度: {structured_feature_dim}")
        print(f"分类类别数: {num_classes}")

        self.model = HybridLSTMClassifier(
            self.bert_model_name,
            structured_feature_dim,
            num_classes,
            hidden_dim=128,  # 减小隐藏层维度
            lstm_layers=1,  # 减少LSTM层数
            dropout_rate=0.3
        ).to(self.device)

        # 优化器
        optimizer = AdamW([
            {'params': [p for p in self.model.bert.parameters() if p.requires_grad], 'lr': learning_rate / 10},
            {'params': self.model.text_lstm.parameters(), 'lr': learning_rate},
            {'params': self.model.struct_fc.parameters(), 'lr': learning_rate},
            {'params': self.model.classifier.parameters(), 'lr': learning_rate}
        ], weight_decay=0.01)

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        best_accuracy = 0
        train_losses = []
        val_accuracies = []

        print("\n开始训练...")
        print("=" * 60)

        for epoch in range(num_epochs):
            start_time = time.time()

            # 训练阶段
            self.model.train()
            total_loss = 0
            batch_count = 0

            # 使用tqdm显示训练进度
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [训练]')

            for batch in train_pbar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                structured_features = batch['structured_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, structured_features)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                batch_count += 1

                # 更新进度条显示当前损失
                train_pbar.set_postfix({
                    '损失': f'{loss.item():.4f}',
                    '平均损失': f'{total_loss / batch_count:.4f}'
                })

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            # 验证阶段
            val_accuracy = self.evaluate(val_loader)
            val_accuracies.append(val_accuracy)

            epoch_time = time.time() - start_time

            print(f"\nEpoch {epoch + 1}/{num_epochs} 完成")
            print(f"训练耗时: {epoch_time:.2f}秒")
            print(f'训练损失: {avg_loss:.4f}')
            print(f'验证集准确率: {val_accuracy:.4f}')

            # 显示当前最佳准确率
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"🎉 新的最佳准确率! 保存模型...")
            else:
                print(f"当前最佳准确率: {best_accuracy:.4f}")

            print('-' * 50)

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth', map_location=device))

        print(f"\n训练完成!")
        print(f"最终验证集最佳准确率: {best_accuracy:.4f}")

        return train_losses, val_accuracies

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        true_labels = []

        # 使用tqdm显示评估进度
        eval_pbar = tqdm(data_loader, desc='[验证]')

        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                structured_features = batch['structured_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, structured_features)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

                # 更新进度条显示当前批次准确率
                batch_acc = accuracy_score(labels.cpu().tolist(), preds.cpu().tolist())
                eval_pbar.set_postfix({'批次准确率': f'{batch_acc:.4f}'})

        return accuracy_score(true_labels, predictions)

    def predict(self, data_loader):
        """预测"""
        self.model.eval()
        predictions = []
        true_labels = []

        # 使用tqdm显示预测进度
        pred_pbar = tqdm(data_loader, desc='[预测]')

        with torch.no_grad():
            for batch in pred_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                structured_features = batch['structured_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, structured_features)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        return predictions, true_labels


def main():
    """主函数"""
    # 初始化分类器
    classifier = SecurityAttackClassifier()

    try:
        # 加载并预处理数据
        texts, structured_features, labels = classifier.load_and_preprocess_data('cybersecurity_attacks.csv')

        # 创建数据加载器
        train_loader, val_loader, test_loader = classifier.create_data_loaders(
            texts, structured_features, labels, batch_size=8  # 减小批大小
        )

        # 训练模型
        print("\n开始训练模型...")
        train_losses, val_accuracies = classifier.train(
            train_loader, val_loader, num_epochs=5, learning_rate=2e-5  # 减少训练轮数
        )

        # 显示训练历史
        print("\n训练历史:")
        for epoch, (loss, acc) in enumerate(zip(train_losses, val_accuracies)):
            print(f"Epoch {epoch + 1}: 损失={loss:.4f}, 验证准确率={acc:.4f}")

        # 测试模型
        print("\n测试模型...")
        test_predictions, test_true = classifier.predict(test_loader)
        test_accuracy = accuracy_score(test_true, test_predictions)

        print(f"\n最终测试准确率: {test_accuracy:.4f}")
        print(f"\n分类报告:")
        print(classification_report(test_true, test_predictions,
                                    target_names=classifier.label_encoder.classes_))

        # 显示混淆矩阵
        cm = confusion_matrix(test_true, test_predictions)
        print(f"\n混淆矩阵:")
        print(cm)

    except FileNotFoundError:
        print("错误: 未找到 'cybersecurity_attacks.csv' 文件")
        print("请确保文件存在于当前目录中")

        # 创建示例数据用于测试
        print("\n创建示例数据用于测试...")
        create_sample_data()

        # 重新运行
        print("\n使用示例数据重新运行...")
        texts, structured_features, labels = classifier.load_and_preprocess_data('cybersecurity_attacks.csv')

        # 创建数据加载器
        train_loader, val_loader, test_loader = classifier.create_data_loaders(
            texts, structured_features, labels, batch_size=8
        )

        # 训练模型
        print("\n开始训练模型...")
        train_losses, val_accuracies = classifier.train(
            train_loader, val_loader, num_epochs=3, learning_rate=2e-5
        )

        # 显示训练历史
        print("\n训练历史:")
        for epoch, (loss, acc) in enumerate(zip(train_losses, val_accuracies)):
            print(f"Epoch {epoch + 1}: 损失={loss:.4f}, 验证准确率={acc:.4f}")

        # 测试模型
        print("\n测试模型...")
        test_predictions, test_true = classifier.predict(test_loader)
        test_accuracy = accuracy_score(test_true, test_predictions)

        print(f"\n最终测试准确率: {test_accuracy:.4f}")

    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data():
    """创建示例网络安全数据"""
    np.random.seed(42)

    # 创建示例数据
    n_samples = 1000

    data = {
        'Payload Data': [
            f"GET /malicious.php?cmd={np.random.choice(['exec', 'download', 'inject'])} HTTP/1.1"
            for _ in range(n_samples)
        ],
        'Packet Length': np.random.randint(50, 1500, n_samples),
        'Protocol': np.random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS'], n_samples),
        'Source Port': np.random.randint(1024, 65535, n_samples),
        'Destination Port': np.random.choice([80, 443, 22, 21, 53], n_samples),
        'Anomaly Score': np.random.uniform(0, 1, n_samples),
        'Severity Level': np.random.choice([1, 2, 3, 4], n_samples),  # 使用数值而不是字符串
        'Flow Type': np.random.choice(['Normal', 'Suspicious', 'Malicious'], n_samples),
        'Action Taken': np.random.choice(['Allow', 'Block', 'Monitor'], n_samples),
        'Attack Type': np.random.choice(
            ['DDoS', 'Malware', 'Phishing', 'Brute Force', 'SQL Injection', 'Normal'],
            n_samples,
            p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
        )
    }

    df = pd.DataFrame(data)
    df.to_csv('cybersecurity_attacks.csv', index=False)
    print("已创建示例数据文件 'cybersecurity_attacks.csv'")


if __name__ == "__main__":
    main()