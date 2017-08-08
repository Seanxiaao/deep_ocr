# deep_ocr
代码使用[deep_ocr](https://github.com/JinpengLI/deep_ocr)的方法对身份证进行识别。

代码应存放在deep_ocr/python文件夹下

执行命令

```
#deep_ocr/python
$python getchar.py -i picture2/idcard2.jpg -t thresh
```

可以得到二值化后的字符，其中 -i 后是需要处理的图片的路径，-t提供了两种写入图片的方式，其中thresh是采用自适应算法后得到的二值化图片，color提供处理前的原图。

line113到line136是对倾斜图片的矫正方法，通过采集左上方5*5的像素来判断身份证的大轮廓(line 115)。故对复杂背景适应性差。

line159到line168是通过mask方法找到身份证上蓝色位置(例如:公民身份证号)，并消除其影响。由于色差，这个方法会产生噪声。代码处理了简单的椒盐噪声

line178到line192通过模板的方法遮盖头像。

line321到line359通过切出轮廓的宽长比来判断部首并合并。

line384设置了自适应求阈值的参数，可以针对情况调整。

最后，如果背景复杂会导致代码算法确定出来轮廓位置和实际情况不符(例如：标记name的地方可能是身份证外框等)。
