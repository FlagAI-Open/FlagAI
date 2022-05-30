# GLM I/O

a) 如下图所示，原文包含6个token，两个区间被屏蔽：第一个区间包含第3个token，第二个区间包含第5个和第6个token。

![results1](img/glm_io_1.png)

b) 将输入分成两个部分： A 部分 (将遮挡区间遮盖掉后的文本)和B部分(被遮挡的区间). 注意所有被遮挡区间的顺序会被重新打乱

![results1](img/glm_io_2.png)

c) GLM的输入和输出，输入包括tokens和2个位置编码

![results1](img/glm_io_3.png)

d) 下图里的自注意力机制既通过遮挡文本实现了自编码， 也在预测遮挡区间内文本的过程里实现了自回归

![results1](img/glm_io_4.png)