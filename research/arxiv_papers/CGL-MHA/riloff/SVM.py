import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# 数据预处理函数
def preprocess_text(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            sentence = lines[i].strip()
            label = int(lines[i + 1].strip())
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels

# 加载训练集和测试集
train_file = './train.txt'
test_file = './test.txt'
train_sentences, train_labels = preprocess_text(train_file)
test_sentences, test_labels = preprocess_text(test_file)

# 使用CountVectorizer将文本转换为词袋模型（Bag of Words）
vectorizer = CountVectorizer(max_features=5000)  # 可以根据需要调整特征数
X_train = vectorizer.fit_transform(train_sentences).toarray()
X_test = vectorizer.transform(test_sentences).toarray()

# 将标签转换为numpy数组
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 创建SVM模型并进行训练
svm_model = SVC(kernel='linear')  # 线性核SVM
svm_model.fit(X_train, y_train)

# 预测测试集结果
y_pred = svm_model.predict(X_test)

# 计算准确率和F1分数
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Test accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
