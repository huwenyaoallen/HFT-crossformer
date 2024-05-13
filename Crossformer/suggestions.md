# 5.13
1. dataloder中去除scale，修改gititem为[T, d] --> 1,具体参考注释意见
2. cross_decoder中的pooling以及输出形状，注意到输入输出是[T, d] --> 1
3. 需要进行instance normlization，可参考https://github.com/ts-kim/RevIN，不过我们只需要norm，不需要denorm