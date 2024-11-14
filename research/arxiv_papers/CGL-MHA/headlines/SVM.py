import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score


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


# SVM 训练和评估函数
def train_and_evaluate_svm(train_file, test_file, max_features=5000, kernel='linear'):
    # 加载训练集和测试集
    train_sentences, train_labels = preprocess_text(train_file)
    test_sentences, test_labels = preprocess_text(test_file)

    # 使用 CountVectorizer 将文本转换为词袋模型
    vectorizer = CountVectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train_sentences).toarray()
    x_test = vectorizer.transform(test_sentences).toarray()

    # 将标签转换为 numpy 数组
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    # 创建 SVM 模型并进行训练
    svm_model = SVC(kernel=kernel)
    svm_model.fit(x_train, y_train)

    # 预测测试集结果
    y_pred = svm_model.predict(x_test)

    # 计算准确率和 F1 分数
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, f1


if __name__ == "__main__":
    train_file = './train.txt'
    test_file = './test.txt'
    train_and_evaluate_svm(train_file, test_file)
