# vision-part-for-ZHONGKONG
This is the vision part for mobile robot based on opencv
the main idea for detection is using color channels and blocking matching

这是大二参加中控杯的视觉部分，串口未完成所以最终未使用。
在visual studio中DEBUGx64编译使用opencv通过颜色和特征匹配判断

识别思路：
雪花，绿色最高可以识别
红牛，黄色最高，少量红色可以识别
乐虎，跟红牛同
网球，黄色最高，无红色可以识别
养乐多，只有红色，可识别
特仑苏，只有少量黄，可识别
ad钙奶，绿色红色一定范围，会跟魔方同
哇哈哈，黄色红色一定范围，会跟魔方同
魔方，无法确认颜色，用特征
木块，单一大量颜色可以识别
ad钙奶，哇哈哈，魔方用surf识别
与哇哈哈，AD钙奶模范匹配，剩余的为魔方

