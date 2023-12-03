
这个程序文件是一个用于训练和测试DecoupledSOLOLight模型的配置文件。它基于另一个配置文件`decoupled_solo_r50_fpn_3x_coco.py`进行了修改。

该模型使用了DecoupledSOLOLightHead作为mask_head，它有80个类别，输入通道数为256，使用了4个堆叠的卷积层，特征通道数为256。模型的下采样步长分别为[8, 8, 16, 32, 32]，尺度范围分别为((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048))，位置尺度为0.2，网格数量分别为[40, 36, 24, 16, 12]，分类下采样索引为0。模型的mask损失使用了DiceLoss，sigmoid激活函数，权重为3.0；分类损失使用了FocalLoss，sigmoid激活函数，gamma为2.0，alpha为0.25，权重为1.0。模型的归一化配置使用了GN，分组数为32。

训练时，首先从文件中加载图像，然后加载带有边界框和掩码的注释。接下来，将图像调整为多个尺度，保持宽高比不变。然后进行随机翻转，归一化处理，填充到32的倍数大小，最后将数据打包成默认格式。训练数据的pipeline配置为`train_pipeline`。

测试时，同样从文件中加载图像，然后进行多尺度翻转增强，调整尺寸，归一化处理，填充到32的倍数大小，最后将图像转换为张量并打包。验证和测试数据的pipeline配置为`test_pipeline`。

最后，数据的配置包括训练数据、验证数据和测试数据的pipeline配置。

#### 5.2 decoupled_solo_r50_fpn_1x_coco.py
