import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 生成虚拟数据
x_train = np.random.random((100,100,100,3))
print(x_train.shape)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20,100,100,3))
y_test = keras.utils.to_categorical(np.random.randint(10,size=(20,1)),num_classes=10)
print(y_train.shape)

# 构建模型
model = Sequential()

# 添加一个卷积层，32个过滤器，3x3的过滤器大小，ReLU激活函数，输入形状为100x100x3
model.add(Conv2D(32, (3,3),activation='relu',input_shape=(100,100,3)))

# 添加另一个卷积层，32个过滤器，3x3的过滤器大小，ReLU激活函数
model.add(Conv2D(32,(3,3),activation='relu'))

# 添加最大池化层，池化窗口大小为2x2
model.add(MaxPooling2D(pool_size=(2,2)))

# 添加Dropout层，丢弃率为0.25
model.add(Dropout(0.25))

# 添加另一个卷积层，64个过滤器，3x3的过滤器大小，ReLU激活函数
model.add(Conv2D(64,(3,3),activation='relu'))

# 添加另一个卷积层，64个过滤器，3x3的过滤器大小，ReLU激活函数
model.add(Conv2D(64,(3,3),activation='relu'))

# 添加最大池化层，池化窗口大小为2x2
model.add(MaxPooling2D(pool_size=(2,2)))

# 添加Dropout层，丢弃率为0.25
model.add(Dropout(0.25))

# flatten层，将多维输入一维化
model.add(Flatten())

# 添加全连接层，512个神经元，ReLU激活函数
model.add(Dense(512,activation='relu'))

# 添加Dropout层，丢弃率为0.5
model.add(Dropout(0.5))

# 添加全连接
model.add(Dense(10,activation='softmax'))

# 编译模型，使用SGD优化器，学习率为0.01，动量为0.9，损失函数为分类交叉熵，评估指标为准确率
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

# 编译模型
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# 训练模型，批次大小为32，训练10个周期
model.fit(x_train,y_train,batch_size=32,epochs=10)

# 评估模型
score = model.evaluate(x_test,y_test,batch_size=32)

# 输出评估结果
print("Test loss:",score[0])
print("Test accuracy:",score[1])

# 预测
y_predict = model.predict(x_test)
print(y_predict)